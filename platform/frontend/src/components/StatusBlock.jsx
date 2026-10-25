export default function StatusBlock({ tone = "neutral", title, message, action }) {
  return (
    <div className={`status status-${tone}`} role={tone === "danger" ? "alert" : "status"}>
      <div className="status-title">{title}</div>
      {message && <div className="status-message">{message}</div>}
      {action && (
        <button type="button" className="btn btn-sm" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
