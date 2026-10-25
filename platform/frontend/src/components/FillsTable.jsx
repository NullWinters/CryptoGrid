// 最多 6 位小数，去掉末尾的 0
const fmtFee = (v) => Number(v).toFixed(6).replace(/\.?0+$/, "");

export default function FillsTable({ fills }) {
  if (!fills || fills.length === 0) {
    return <div className="empty-state">暂无成交</div>;
  }

  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label="成交明细">
      <table className="table">
        <thead>
          <tr>
            <th>时间</th>
            <th>标的</th>
            <th>方向</th>
            <th className="num">价格</th>
            <th className="num">数量</th>
            <th className="num">手续费</th>
          </tr>
        </thead>
        <tbody>
          {[...fills].reverse().map((f, i) => {
            const buy = f.side === "buy";
            return (
              <tr key={`${f.time}-${f.coin}-${i}`}>
                <td className="mono">{new Date(f.time).toLocaleTimeString()}</td>
                <td>{f.coin}</td>
                <td className={buy ? "up" : "down"}>{buy ? "买入" : "卖出"}</td>
                <td className={`num ${buy ? "up" : "down"}`}>{f.px}</td>
                <td className={`num ${buy ? "up" : "down"}`}>{f.sz}</td>
                <td className="num muted">{fmtFee(f.fee)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
