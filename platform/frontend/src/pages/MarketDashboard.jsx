import React, { useEffect, useRef, useState } from "react";
import { fetchCandles, fetchInstruments, openStream } from "../api.js";
import CandleChart from "../components/CandleChart.jsx";
import Icon from "../components/Icon.jsx";
import OrderBook from "../components/OrderBook.jsx";
import Panel from "../components/Panel.jsx";
import Skeleton from "../components/Skeleton.jsx";
import StatusBlock from "../components/StatusBlock.jsx";
import TradesList from "../components/TradesList.jsx";

const INTERVALS = [
  "1m", "3m", "5m", "15m", "30m",
  "1h", "2h", "4h", "8h", "12h",
  "1d", "3d", "1w", "1M",
];

const STATUS_TEXT = {
  connecting: "连接中",
  connected: "已连接",
  disconnected: "已断开",
  reconnecting: "重连中",
};

export default function MarketDashboard({ palette: p, convention, theme, onConventionChange, onThemeChange }) {
  const [env, setEnv] = useState("mainnet"); // 看板环境与策略页互相独立
  const [instruments, setInstruments] = useState([]);
  const [coin, setCoin] = useState("BTC");
  const [interval, setIntervalVal] = useState("1h");
  const [candles, setCandles] = useState([]);
  const [book, setBook] = useState(null);
  const [trades, setTrades] = useState([]);
  const [status, setStatus] = useState("connecting");
  const [error, setError] = useState("");
  const [instrumentsLoading, setInstrumentsLoading] = useState(true);
  const [instrumentsError, setInstrumentsError] = useState("");
  const [candlesLoading, setCandlesLoading] = useState(true);
  const [candlesError, setCandlesError] = useState("");

  const streamRef = useRef(null);
  const liveCandle = useRef(null);
  const sel = useRef({ env, coin, interval });
  sel.current = { env, coin, interval };

  function loadCandles(e, c, iv) {
    setCandlesLoading(true);
    setCandlesError("");
    return fetchCandles(e, c, iv)
      .then((rows) => {
        setCandles(rows);
        setCandlesLoading(false);
      })
      .catch(() => {
        setCandlesError("无法获取 K 线数据，请检查后端与网络");
        setCandlesLoading(false);
      });
  }

  function loadInstruments(e) {
    setInstrumentsLoading(true);
    setInstrumentsError("");
    return fetchInstruments(e)
      .then((list) => {
        setInstruments(list);
        setInstrumentsLoading(false);
      })
      .catch(() => {
        setInstrumentsError("无法获取标的列表，请检查后端与网络");
        setInstrumentsLoading(false);
      });
  }

  useEffect(() => {
    const stream = openStream({
      candle: (d) => liveCandle.current?.(d),
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
          setCandlesError("");
          setCandlesLoading(false);
        } else if (d.state === "connected") {
          setError("");
          const s = sel.current;
          loadCandles(s.env, s.coin, s.interval);
        }
      },
    });
    streamRef.current = stream;
    return () => stream.close();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadInstruments(env);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env]);

  useEffect(() => {
    reload(env, coin, interval);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env, coin, interval]);

  function reload(e, c, iv) {
    setTrades([]);
    setBook(null);
    setCandles([]);
    loadCandles(e, c, iv);
    streamRef.current?.subscribe(e, c, iv);
  }

  const available = instruments.some((i) => i.name === coin);
  const bids = book?.levels?.[0] ?? [];
  const asks = book?.levels?.[1] ?? [];
  const chartState = (
    <div className="panel-state">
      {candlesError ? (
        <StatusBlock
          tone="danger"
          title="K 线加载失败"
          message={candlesError}
          action={{ label: "重试", onClick: () => loadCandles(env, coin, interval) }}
        />
      ) : candlesLoading ? (
        <Skeleton variant="chart" />
      ) : candles.length === 0 ? (
        <StatusBlock title="暂无 K 线数据" message="连接中断或该标的当前没有成交" />
      ) : null}
    </div>
  );

  return (
    <div className="page market-page">
      <header className="page-header">
        <h1 className="page-title">行情看板</h1>
        <div className="control-bar">
          <div className="control-group">
            <label className="control">
              环境
              <select value={env} onChange={(e) => setEnv(e.target.value)}>
                <option value="mainnet">主网</option>
                <option value="testnet">测试网</option>
              </select>
            </label>
            <label className="control">
              标的
              <select value={coin} onChange={(e) => setCoin(e.target.value)}>
                {instruments.map((i) => (
                  <option key={i.name} value={i.name}>
                    {i.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="control">
              周期
              <select value={interval} onChange={(e) => setIntervalVal(e.target.value)}>
                {INTERVALS.map((iv) => (
                  <option key={iv} value={iv}>
                    {iv}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <div className="control-group right">
            <span className={status === "connected" ? "small up" : "small muted"}>{STATUS_TEXT[status] ?? status}</span>
            <label className="control">
              展示习惯
              <select value={convention} onChange={(e) => onConventionChange(e.target.value)}>
                <option value="greenUp">绿涨红跌</option>
                <option value="redUp">红涨绿跌</option>
              </select>
            </label>
            <button
              type="button"
              className="btn"
              onClick={() => onThemeChange(theme === "light" ? "dark" : "light")}
              aria-label={`切换到${theme === "light" ? "暗色" : "亮色"}主题`}
            >
              <Icon name="theme" />
              {theme === "light" ? "暗色" : "亮色"}
            </button>
          </div>
        </div>
      </header>

      {error && (
        <div className="banner banner-warning" role="alert">
          <span className="banner-body">
            <Icon name="warning" /> {error}
          </span>
        </div>
      )}

      {instrumentsError && (
        <div className="panel-state">
          <StatusBlock
            tone="danger"
            title="标的列表加载失败"
            message={instrumentsError}
            action={{ label: "重试", onClick: () => loadInstruments(env) }}
          />
        </div>
      )}

      {!instrumentsError && !instrumentsLoading && !available && (
        <div className="banner banner-warning" role="alert">
          <span className="banner-body">
            <Icon name="warning" /> 标的 {coin} 在当前环境不可用，请重新选择
          </span>
        </div>
      )}

      {instrumentsLoading && (
        <div className="market-grid">
          <div className="market-left">
            <Panel title="行情">
              <Skeleton variant="chart" />
            </Panel>
            <div className="book-grid">
              <Panel title="买单簿">
                <Skeleton rows={6} />
              </Panel>
              <Panel title="卖单簿">
                <Skeleton rows={6} />
              </Panel>
            </div>
          </div>
          <Panel title="交易记录" className="panel-fill trades-panel">
            <Skeleton rows={12} />
          </Panel>
        </div>
      )}

      {!instrumentsError && !instrumentsLoading && available && (
        <div className="market-grid">
          <div className="market-left">
            <Panel title={`${coin} · ${interval}`} className="panel-chart">
              {candlesError || candlesLoading || candles.length === 0 ? (
                chartState
              ) : (
                <CandleChart
                  candles={candles}
                  liveRef={liveCandle}
                  up={p.up}
                  down={p.down}
                  bg={p.bg}
                  fg={p.fg}
                  border={p.border}
                />
              )}
            </Panel>
            <div className="book-grid">
              <Panel title="买单簿">
                <OrderBook levels={bids} side="buy" label="买单簿" />
              </Panel>
              <Panel title="卖单簿">
                <OrderBook levels={asks} side="sell" label="卖单簿" />
              </Panel>
            </div>
          </div>
          <Panel title="交易记录" className="panel-fill trades-panel">
            {candlesLoading && trades.length === 0 ? (
              <Skeleton rows={12} />
            ) : trades.length === 0 ? (
              <div className="panel-state">
                <StatusBlock title="暂无最新成交" message="连接中断或该标的当前没有成交" />
              </div>
            ) : (
              <TradesList trades={trades} />
            )}
          </Panel>
        </div>
      )}
    </div>
  );
}
