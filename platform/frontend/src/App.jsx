import React, { useEffect, useRef, useState } from "react";
import { fetchInstruments, fetchCandles, openStream } from "./api.js";
import { palette } from "./theme.js";
import CandleChart from "./components/CandleChart.jsx";
import OrderBook from "./components/OrderBook.jsx";
import TradesList from "./components/TradesList.jsx";

const INTERVALS = [
  "1m", "3m", "5m", "15m", "30m",
  "1h", "2h", "4h", "8h", "12h",
  "1d", "3d", "1w", "1M",
];

function Panel({ title, border, fg, children, style }) {
  return (
    <div style={{ border: `1px solid ${border}`, borderRadius: 4, padding: 8, ...style }}>
      <div style={{ color: fg, fontSize: 13, marginBottom: 6, fontWeight: 600 }}>{title}</div>
      {children}
    </div>
  );
}

export default function App() {
  const [env, setEnv] = useState("mainnet");
  const [instruments, setInstruments] = useState([]);
  const [coin, setCoin] = useState("BTC");
  const [interval, setIntervalVal] = useState("1h");
  const [candles, setCandles] = useState([]);
  const [book, setBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [status, setStatus] = useState("connecting");
  const [error, setError] = useState("");
  const [convention, setConvention] = useState("greenUp"); // FR-001 默认绿涨红跌
  const [theme, setTheme] = useState("light"); // FR-003 默认亮色

  const p = palette(convention, theme);

  const streamRef = useRef(null);
  const liveCandle = useRef(null); // 最新一根的实时更新回调传给 CandleChart
  const sel = useRef({ env, coin, interval });
  sel.current = { env, coin, interval }; // 供 WS 回调读取最新选择，避免闭包取到初始值

  // 整页背景与文字随主题变化（FR-004）
  useEffect(() => {
    document.body.style.background = p.bg;
    document.body.style.color = p.fg;
  }, [p.bg, p.fg]);

  // 打开 WS，注册各频道回调。仅挂载一次。
  useEffect(() => {
    const stream = openStream({
      candle: (d) => {
        if (liveCandle.current) liveCandle.current(d);
      },
      l2Book: (d) => setBook(d),
      trades: (d) =>
        setTrades((prev) => {
          const incoming = Array.isArray(d) ? d : [d];
          return [...incoming.reverse(), ...prev].slice(0, 50);
        }),
      status: (d) => {
        setStatus(d.state);
        if (d.state === "disconnected") {
          setError("与 Hyperliquid 的连接已中断，正在重连…");
          setCandles([]);
          setBook(null);
          setTrades([]);
        } else if (d.state === "connected") {
          setError("");
          const s = sel.current;
          fetchCandles(s.env, s.coin, s.interval)
            .then(setCandles)
            .catch(() => setError("无法获取 K 线数据"));
        }
      },
    });
    streamRef.current = stream;
    return () => stream.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    fetchInstruments(env)
      .then((list) => {
        if (!cancelled) setInstruments(list);
      })
      .catch(() => !cancelled && setError("无法获取标的列表"));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env]);

  // 仅 env/coin/interval 变化时重订阅；convention 与 theme 变化不在此列（FR-009）
  useEffect(() => {
    reload(env, coin, interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env, coin, interval]);

  function reload(e, c, iv) {
    setTrades([]);
    setBook(null);
    fetchCandles(e, c, iv)
      .then((rows) => setCandles(rows))
      .catch(() => setError("无法获取 K 线数据"));
    if (streamRef.current) streamRef.current.subscribe(e, c, iv);
  }

  const available = instruments.some((i) => i.name === coin);
  const bids = book?.levels?.[0] ?? [];
  const asks = book?.levels?.[1] ?? [];

  return (
    <div style={{ fontFamily: "sans-serif", padding: 16, background: p.bg, color: p.fg, minHeight: "100vh" }}>
      <header style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12 }}>
        <h1 style={{ fontSize: 18, margin: 0 }}>行情看板</h1>
        <label>
          环境{" "}
          <select value={env} onChange={(e) => setEnv(e.target.value)}>
            <option value="mainnet">主网</option>
            <option value="testnet">测试网</option>
          </select>
        </label>
        <label>
          标的{" "}
          <select value={coin} onChange={(e) => setCoin(e.target.value)}>
            {instruments.map((i) => (
              <option key={i.name} value={i.name}>
                {i.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          周期{" "}
          <select value={interval} onChange={(e) => setIntervalVal(e.target.value)}>
            {INTERVALS.map((iv) => (
              <option key={iv} value={iv}>
                {iv}
              </option>
            ))}
          </select>
        </label>
        <label>
          展示习惯{" "}
          <select value={convention} onChange={(e) => setConvention(e.target.value)}>
            <option value="greenUp">绿涨红跌</option>
            <option value="redUp">红涨绿跌</option>
          </select>
        </label>
        <button onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}>
          {theme === "light" ? "亮色" : "暗色"}
        </button>
        <span style={{ color: status === "connected" ? p.up : p.muted }}>{status}</span>
      </header>

      {error && (
        <div style={{ background: "#fde8e8", color: "#b91c1c", padding: 8, marginBottom: 12 }}>{error}</div>
      )}

      {!available && instruments.length > 0 && (
        <div style={{ background: "#fef3c7", color: "#92400e", padding: 8, marginBottom: 12 }}>
          标的 {coin} 在当前环境不可用，请重新选择
        </div>
      )}

      {available && (
        <div style={{ display: "flex", gap: 16, alignItems: "stretch" }}>
          <div style={{ flex: 2, display: "flex", flexDirection: "column", gap: 16 }}>
            <CandleChart candles={candles} liveRef={liveCandle} up={p.up} down={p.down} bg={p.bg} fg={p.fg} border={p.border} />
            <div style={{ display: "flex", gap: 16 }}>
              <Panel title="买单簿" border={p.border} fg={p.fg} style={{ flex: 1 }}>
                <OrderBook levels={bids} color={p.up} muted={p.muted} />
              </Panel>
              <Panel title="卖单簿" border={p.border} fg={p.fg} style={{ flex: 1 }}>
                <OrderBook levels={asks} color={p.down} muted={p.muted} />
              </Panel>
            </div>
          </div>
          <Panel title="交易记录" border={p.border} fg={p.fg} style={{ flex: 1 }}>
            <TradesList trades={trades} up={p.up} down={p.down} muted={p.muted} />
          </Panel>
        </div>
      )}
    </div>
  );
}
