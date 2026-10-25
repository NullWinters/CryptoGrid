const cell = "num";

const fmt = (v, digits) =>
  v == null ? null : typeof v === "number" && digits != null ? v.toFixed(digits) : String(v);

export default function RunningStatusBar({ positions, onRetry }) {
  if (!positions || positions.length === 0) {
    return <div className="empty-state">暂无持仓</div>;
  }

  const value = (v, digits, failed) => {
    const text = fmt(v, digits);
    return text === null ? <span className="muted">{failed ? "建仓失败" : "不可用"}</span> : text;
  };

  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label="运行状态">
      <table className="table">
        <thead>
          <tr>
            <th>标的</th>
            <th className={cell}>数量</th>
            <th className={cell}>入场均价</th>
            <th className={cell}>未实现盈亏</th>
            <th className={cell}>保证金占用</th>
            <th className={cell}>强平价格</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {positions.map((p, i) => {
            const failed = p.opened === 0;
            return (
              <tr key={p.coin ?? i}>
                <td>{p.coin}</td>
                <td className={cell}>{value(p.sz, null, failed)}</td>
                <td className={cell}>{value(p.entry_px, null, failed)}</td>
                <td className={cell}>
                  {p.unrealized_pnl == null ? (
                    value(null, 2, failed)
                  ) : (
                    <span className={p.unrealized_pnl >= 0 ? "up" : "down"}>{fmt(p.unrealized_pnl, 2)}</span>
                  )}
                </td>
                <td className={cell}>{value(p.margin_used, 2, failed)}</td>
                <td className={cell}>{value(p.liquidation_px, null, failed)}</td>
                <td>
                  {failed && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      disabled={!onRetry}
                      onClick={() => onRetry?.(p.coin)}
                    >
                      重试
                    </button>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
