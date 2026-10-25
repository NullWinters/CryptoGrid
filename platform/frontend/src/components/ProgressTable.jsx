const STATUS = { pending: "待提交", submitting: "提交中", filled: "已成交", failed: "失败" };
const ACTION = { open: "开仓", close: "平仓" };

export default function ProgressTable({ items }) {
  if (!items || items.length === 0) {
    return <div className="empty-state">暂无进行中的操作</div>;
  }

  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label="执行进度">
      <table className="table">
        <thead>
          <tr>
            <th>标的</th>
            <th>动作</th>
            <th>状态</th>
            <th className="num">数量</th>
            <th className="num">成交价</th>
            <th>说明</th>
          </tr>
        </thead>
        <tbody>
          {items.map((item, i) => (
            <tr key={`${item.coin}-${i}`}>
              <td>{item.coin}</td>
              <td>{ACTION[item.action] ?? item.action}</td>
              <td className={item.status === "filled" ? "up" : item.status === "failed" ? "down" : ""}>
                {STATUS[item.status] ?? item.status}
              </td>
              <td className="num">{item.sz ?? "不可用"}</td>
              <td className="num">{item.px ?? "不可用"}</td>
              <td className="muted">{item.error ?? "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
