const BASE = "/api/strategy";

async function request(path, options) {
  const res = await fetch(BASE + path, options);
  const text = await res.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!res.ok) {
    const detail = body && body.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail != null
          ? JSON.stringify(detail)
          : `请求失败 ${res.status}`;
    const err = new Error(message);
    err.body = body;
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : body;
}

function post(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function fetchAccount(env) {
  return request(`/account?env=${encodeURIComponent(env)}`);
}

export function fetchCurrent() {
  return request("/current");
}

export function startStrategy(payload) {
  return post("/start", payload);
}

export function retryPosition(coin) {
  return post("/retry", { coin });
}

export function stopStrategy() {
  return post("/stop", {});
}

export function resolveStrategy(action) {
  return post("/resolve", { action });
}

export function fetchEquity() {
  return request("/equity");
}

export function fetchFills(strategyId) {
  return request(
    typeof strategyId === "number"
      ? `/fills?strategy_id=${strategyId}`
      : "/fills",
  );
}

export function fetchLogs() {
  return request("/logs");
}

export function fetchProgress() {
  return request("/progress");
}
