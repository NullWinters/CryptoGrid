import React, { useEffect, useRef, useState } from "react";
import { fetchInstruments, openStream } from "../api.js";
import {
  fetchAccount,
  fetchCurrent,
  fetchEquity,
  fetchFills,
  fetchLogs,
  fetchProgress,
  resolveStrategy,
  retryPosition,
  startStrategy,
  stopStrategy,
} from "../strategyApi.js";
import EquityChart from "../components/EquityChart.jsx";
import FillsTable from "../components/FillsTable.jsx";
import Icon from "../components/Icon.jsx";
import Modal from "../components/Modal.jsx";
import Panel from "../components/Panel.jsx";
import ProgressTable from "../components/ProgressTable.jsx";
import RunningStatusBar from "../components/RunningStatusBar.jsx";
import Skeleton from "../components/Skeleton.jsx";
import StatusBlock from "../components/StatusBlock.jsx";

const BENCHMARKS = [
  { value: "btc", label: "BTC 买入持有" },
  { value: "eth", label: "ETH 买入持有" },
  { value: "paxg", label: "PAXG 买入持有" },
  { value: "earn", label: "USDC 生息" },
];

const STATUS_ANNOUNCE = {
  running: "策略运行中",
  mismatch: "对账不一致，策略已暂停",
  closed: "策略已终止",
  open_failed: "策略建仓失败",
};

const emptyRow = () => ({ coin: "BTC", amount_usdc: "", leverage: "1" });

export default function StrategyTrading({ palette: p, convention, theme, onConventionChange, onThemeChange }) {
  const [env, setEnv] = useState("testnet"); // 策略环境与看板独立，默认测试网更安全
  const [benchmark, setBenchmark] = useState("btc");
  const [rows, setRows] = useState([emptyRow()]);
  const [instruments, setInstruments] = useState([]);
  const [instrumentsError, setInstrumentsError] = useState("");
  const [current, setCurrent] = useState(null);
  const [currentError, setCurrentError] = useState("");
  const [equity, setEquity] = useState({ points: [], total_invested: 0, max_drawdown: 0, offline_ranges: [] });
  const [equityError, setEquityError] = useState("");
  const [fills, setFills] = useState([]);
  const [logs, setLogs] = useState([]);
  const [unit, setUnit] = useState("abs");
  const [errors, setErrors] = useState([]);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [fillsModal, setFillsModal] = useState(false);
  const [modalFills, setModalFills] = useState([]);
  const [balance, setBalance] = useState(null);
  const [balanceError, setBalanceError] = useState("");
  const [progress, setProgress] = useState(null);
  const [progressOpen, setProgressOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);
  const [announcement, setAnnouncement] = useState("");

  const streamRef = useRef(null);
  const lastId = useRef(null);
  const stoppedId = useRef(null);
  const pollRef = useRef(null);
  const pendingFills = useRef(false);
  const prevStatus = useRef(undefined);

  const running = current?.status === "running";
  const mismatch = current?.status === "mismatch";
  const locked = running || mismatch;

  async function refreshData() {
    const [eq, lg, cur] = await Promise.allSettled([fetchEquity(), fetchLogs(), fetchCurrent()]);

    if (eq.status === "fulfilled") {
      setEquity(eq.value);
      setEquityError("");
    } else {
      setEquityError("无法获取净值数据");
    }

    if (lg.status === "fulfilled") setLogs(lg.value);

    if (cur.status === "fulfilled") {
      const strategy = cur.value;
      setCurrent(strategy);
      setCurrentError("");
      if (strategy) {
        lastId.current = strategy.id;
        const rowsFills = await fetchFills(strategy.id).catch(() => []);
        setFills(rowsFills);
      } else {
        setFills([]);
      }
    } else {
      setCurrentError("无法获取当前策略");
    }

    setInitialLoading(false);
  }

  useEffect(() => {
    const stream = openStream({ strategy: () => refreshData() });
    streamRef.current = stream;
    const openTimer = setTimeout(() => stream.watchStrategy(), 500);
    refreshData();
    return () => {
      clearTimeout(openTimer);
      stream.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    loadInstruments(env);
    loadBalance(env);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [env]);

  function loadInstruments(e) {
    setInstrumentsError("");
    fetchInstruments(e)
      .then(setInstruments)
      .catch(() => setInstrumentsError("无法获取标的列表"));
  }

  function loadBalance(e) {
    setBalanceError("");
    fetchAccount(e)
      .then((account) => setBalance(account.withdrawable))
      .catch(() => {
        setBalance(null);
        setBalanceError("无法获取可用余额");
      });
  }

  useEffect(() => {
    if (!current) {
      if (prevStatus.current !== undefined) {
        prevStatus.current = undefined;
        setAnnouncement("策略已终止");
      }
      return;
    }
    if (prevStatus.current !== current.status) {
      prevStatus.current = current.status;
      const label = STATUS_ANNOUNCE[current.status] ?? current.status;
      setAnnouncement(current.note ? `${label}。${current.note}` : label);
    }
  }, [current]);

  useEffect(() => {
    if (errors.length > 0) setAnnouncement(errors.join("；"));
  }, [errors]);

  function pushErrors(list) {
    setErrors(Array.isArray(list) ? list : [String(list)]);
  }

  function refreshBalance() {
    loadBalance(env);
  }

  useEffect(() => () => clearInterval(pollRef.current), []);

  function startPolling() {
    clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const snap = await fetchProgress();
        setProgress(snap);
        if (snap?.finished) clearInterval(pollRef.current);
      } catch {
        clearInterval(pollRef.current);
      }
    }, 300);
  }

  // 每次操作在这里把弹窗打开一次，后续刷新不会重新打开
  async function withProgress(fn) {
    setBusy(true);
    setProgress(null);
    setProgressOpen(true);
    startPolling();
    try {
      return await fn();
    } finally {
      clearInterval(pollRef.current);
      fetchProgress().then(setProgress).catch(() => {});
      setBusy(false);
    }
  }

  function closeProgress() {
    setProgressOpen(false);
    if (pendingFills.current) {
      pendingFills.current = false;
      setFillsModal(true);
    }
  }

  function toPayload() {
    return {
      env,
      benchmark,
      positions: rows
        .filter((row) => row.coin)
        .map((row) => ({
          coin: row.coin,
          amount_usdc: Number(row.amount_usdc),
          leverage: Number(row.leverage),
        })),
    };
  }

  async function onStart() {
    setErrors([]);
    try {
      const strategy = await withProgress(() => startStrategy(toPayload()));
      setCurrent(strategy);
      lastId.current = strategy.id;
      refreshData();
    } catch (err) {
      if (err.status === 400) setProgressOpen(false);
      const detail = err.body?.detail;
      pushErrors(Array.isArray(detail) ? detail : err.message);
    }
  }

  async function onRetry(coin) {
    setErrors([]);
    try {
      await withProgress(() => retryPosition(coin));
      refreshData();
    } catch (err) {
      if (err.status === 400) setProgressOpen(false);
      const detail = err.body?.detail;
      pushErrors(Array.isArray(detail) ? detail : err.message);
    }
  }

  async function onConfirmStop() {
    setConfirmOpen(false);
    const id = current?.id;
    const closable = (current?.positions ?? []).some((position) => position.opened && Number(position.sz) > 0);
    setErrors([]);
    try {
      if (closable) {
        await withProgress(() => stopStrategy());
      } else {
        // 无持仓可平：不弹进度，直接终止并用横幅说明
        await stopStrategy();
        pushErrors(["本轮无持仓可平，策略已标记为已终止"]);
      }
      stoppedId.current = id ?? lastId.current;
      refreshData();
      if (stoppedId.current != null) {
        const rowsFills = await fetchFills(stoppedId.current).catch(() => []);
        setModalFills(rowsFills);
        if (closable) {
          pendingFills.current = true; // 进度弹窗关闭后再弹成交明细
        } else {
          setFillsModal(true);
        }
      }
    } catch (err) {
      const detail = err.body?.detail;
      pushErrors(Array.isArray(detail) ? detail : err.message);
    }
  }

  async function onResolve(action) {
    setErrors([]);
    try {
      await resolveStrategy(action);
      refreshData();
    } catch (err) {
      pushErrors(err.message);
    }
  }

  const totalPreview = rows.reduce((sum, row) => sum + (Number(row.amount_usdc) || 0), 0);
  const progressTitle =
    progress?.op === "stop" ? "平仓进度" : progress?.op === "retry" ? "重试进度" : "下单进度";
  const chartColors = {
    strategy: p.accentText,
    benchmark: p.muted,
    danger: p.danger,
    bg: p.bg,
    fg: p.fg,
    border: p.border,
  };

  return (
    <div className="page">
      <div className="sr-only" role="status" aria-live="polite">
        {announcement}
      </div>

      <header className="page-header">
        <h1 className="page-title">策略交易</h1>
        <div className="control-bar">
          <div className="control-group">
            <label className="control">
              环境
              <select value={env} disabled={locked} onChange={(e) => setEnv(e.target.value)}>
                <option value="mainnet">主网</option>
                <option value="testnet">测试网</option>
              </select>
            </label>
            <span className="balance-wrap">
              <span className="balance">
                可用余额 <strong>{balance != null ? `${balance.toFixed(2)} USDC` : "不可用"}</strong>
              </span>
              <button type="button" className="icon-btn" onClick={refreshBalance} aria-label="刷新可用余额" title="刷新可用余额">
                <Icon name="refresh" />
              </button>
            </span>
            {balanceError && <span className="small down">{balanceError}</span>}
            {locked && <span className="small muted">运行期间环境已锁定</span>}
          </div>
          <div className="control-group right">
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

      {errors.length > 0 && (
        <div className="banner banner-danger" role="alert">
          <div className="banner-body">
            <div className="banner-title">操作未完成</div>
            <ul className="banner-list">
              {errors.map((message, i) => (
                <li key={i}>{message}</li>
              ))}
            </ul>
          </div>
          <button type="button" className="icon-btn" onClick={() => setErrors([])} aria-label="关闭错误提示">
            <Icon name="close" />
          </button>
        </div>
      )}

      {mismatch && (
        <div className="banner banner-danger" role="alert">
          <div className="banner-body">
            <div className="banner-title">对账不一致，策略已暂停</div>
            <div className="small">{current.note}</div>
            <div className="mismatch-actions">
              <button type="button" className="btn" onClick={() => onResolve("resume")}>按实际持仓继续</button>
              <button type="button" className="btn btn-danger" onClick={() => onResolve("close")}>平仓终止</button>
            </div>
          </div>
        </div>
      )}

      {currentError && (
        <div className="panel-state">
          <StatusBlock
            tone="danger"
            title="当前策略加载失败"
            message={currentError}
            action={{ label: "重试", onClick: refreshData }}
          />
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

      <div className="strategy-layout">
        <aside className="sidebar" aria-label="策略列表">
          <button type="button" className="side-btn active">Buy &amp; Hold</button>
          <button type="button" className="side-btn" disabled>更多策略 敬请期待</button>
        </aside>

        <main className="strategy-main">
          <Panel
            title="净值曲线"
            actions={
              <label className="control">
                纵轴
                <select value={unit} onChange={(e) => setUnit(e.target.value)}>
                  <option value="abs">绝对金额</option>
                  <option value="pct">收益率百分比</option>
                </select>
              </label>
            }
          >
            {initialLoading ? (
              <Skeleton variant="chart" />
            ) : equityError ? (
              <StatusBlock
                tone="danger"
                title="净值加载失败"
                message={equityError}
                action={{ label: "重试", onClick: refreshData }}
              />
            ) : equity.points.length === 0 ? (
              <div className="equity-placeholder">
                <StatusBlock title="尚未运行策略" message="运行后在此显示组合净值与基准" />
              </div>
            ) : (
              <EquityChart
                points={equity.points}
                totalInvested={equity.total_invested}
                maxDrawdown={equity.max_drawdown}
                offlineRanges={equity.offline_ranges}
                unit={unit}
                colors={chartColors}
              />
            )}
          </Panel>

          <Panel title="运行状态">
            {initialLoading ? (
              <Skeleton rows={3} />
            ) : locked ? (
              <RunningStatusBar
                positions={current.positions}
                onRetry={running && !busy ? onRetry : undefined}
              />
            ) : (
              <StatusBlock title="尚未运行策略" message="配置标的与金额后点击运行" />
            )}
          </Panel>

          <Panel title="交易操作">
            <div className="row">
              <label className="control">
                基准
                <select value={benchmark} disabled={locked} onChange={(e) => setBenchmark(e.target.value)}>
                  {BENCHMARKS.map((item) => (
                    <option key={item.value} value={item.value}>{item.label}</option>
                  ))}
                </select>
              </label>
              {locked && <span className="small muted">基准在运行期间锁定</span>}
            </div>

            <fieldset disabled={locked} style={{ border: "none", padding: 0, margin: "10px 0 0" }}>
              <div className="rows">
                {rows.map((row, i) => (
                  <div className="row" key={i}>
                    <select
                      className="coin"
                      value={row.coin}
                      onChange={(e) => setRows(rows.map((item, j) => (j === i ? { ...item, coin: e.target.value } : item)))}
                    >
                      {instruments.length === 0 && <option value={row.coin}>{row.coin}</option>}
                      {instruments.map((inst) => (
                        <option key={inst.name} value={inst.name}>{inst.name}</option>
                      ))}
                    </select>
                    <input
                      className="amount"
                      type="number"
                      placeholder="金额 USDC"
                      value={row.amount_usdc}
                      onChange={(e) => setRows(rows.map((item, j) => (j === i ? { ...item, amount_usdc: e.target.value } : item)))}
                    />
                    <input
                      className="lev"
                      type="number"
                      min="1"
                      placeholder="杠杆"
                      value={row.leverage}
                      onChange={(e) => setRows(rows.map((item, j) => (j === i ? { ...item, leverage: e.target.value } : item)))}
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setRows(rows.filter((_, j) => j !== i))}
                      disabled={rows.length === 1}
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
              <button type="button" className="btn btn-sm" onClick={() => setRows([...rows, emptyRow()])}>
                添加标的
              </button>
            </fieldset>

            <div className="row-actions">
              <button type="button" className="btn btn-primary" onClick={onStart} disabled={locked || busy}>
                运行
              </button>
              <button type="button" className="btn btn-danger" onClick={() => setConfirmOpen(true)} disabled={!running || busy}>
                终止
              </button>
            </div>

            <div className="preview">
              {running ? (
                <div className="fills-inline">
                  <FillsTable fills={fills} />
                </div>
              ) : (
                <>
                  <div className="preview-title">执行预览</div>
                  {rows.length === 0 ? (
                    <div className="empty-state">至少添加一个标的</div>
                  ) : (
                    <div className="table-scroll">
                      <table className="table">
                        <thead>
                          <tr>
                            <th>标的</th>
                            <th className="num">金额</th>
                            <th className="num">杠杆</th>
                            <th className="num">名义价值</th>
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row, i) => (
                            <tr key={i}>
                              <td>{row.coin}</td>
                              <td className="num">{Number(row.amount_usdc) || 0}</td>
                              <td className="num">{Number(row.leverage) || 0}</td>
                              <td className="num">
                                {((Number(row.amount_usdc) || 0) * (Number(row.leverage) || 0)).toFixed(2)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                  <div className="preview-total">合计投入 {totalPreview.toFixed(2)} USDC</div>
                </>
              )}
            </div>
          </Panel>

          {logs.length > 0 && (
            <Panel title="运行日志">
              <div className="logs">
                {logs.slice(0, 5).map((log, i) => (
                  <div key={i}>
                    {new Date(log.time).toLocaleTimeString()} [{log.level}] {log.message}
                  </div>
                ))}
              </div>
            </Panel>
          )}
        </main>
      </div>

      <Modal open={progressOpen} title={progressTitle} onClose={closeProgress} wide>
        <ProgressTable items={progress?.items} />
      </Modal>

      <Modal
        open={confirmOpen && !!current}
        title="确认终止策略"
        onClose={() => setConfirmOpen(false)}
        footer={
          <>
            <button type="button" className="btn" onClick={() => setConfirmOpen(false)}>取消</button>
            <button type="button" className="btn btn-danger-solid" onClick={onConfirmStop}>确认终止</button>
          </>
        }
      >
        <div className="small modal-desc">将以市价平掉以下持仓：</div>
        <ul className="banner-list">
          {current?.positions.filter((position) => position.opened).map((position) => (
            <li key={position.coin}>{position.coin} 数量 {position.sz}</li>
          ))}
        </ul>
      </Modal>

      <Modal open={fillsModal} title="本轮成交明细" onClose={() => setFillsModal(false)} wide>
        <FillsTable fills={modalFills} />
      </Modal>
    </div>
  );
}
