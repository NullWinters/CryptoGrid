// 后端 REST 与 WebSocket 访问封装。
// 后端合约见 specs/001-realtime-market-dashboard/contracts/stream.md

export async function fetchInstruments(env) {
  const res = await fetch(`/api/instruments?env=${encodeURIComponent(env)}`);
  if (!res.ok) throw new Error(`instruments ${res.status}`);
  return res.json();
}

export async function fetchCandles(env, coin, interval) {
  const q = `env=${encodeURIComponent(env)}&coin=${encodeURIComponent(coin)}&interval=${encodeURIComponent(interval)}`;
  const res = await fetch(`/api/candles?${q}`);
  if (!res.ok) throw new Error(`candles ${res.status}`);
  return res.json();
}

// 维护一条到 /api/stream 的 WebSocket，按 channel 分发消息。
// handlers: { candle, l2Book, trades, status }，每个为 (data) => void
export function openStream(handlers) {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
  let current = null;
  let watching = false;

  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    const fn = handlers[msg.channel];
    if (fn) fn(msg.data);
  };
  ws.onopen = () => {
    if (current) ws.send(JSON.stringify({ action: "subscribe", ...current }));
    if (watching) ws.send(JSON.stringify({ action: "watch_strategy" }));
  };

  return {
    subscribe(env, coin, interval) {
      current = { env, coin, interval };
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "subscribe", ...current }));
      }
    },
    watchStrategy() {
      watching = true;
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "watch_strategy" }));
      }
    },
    close() {
      ws.close();
    },
  };
}
