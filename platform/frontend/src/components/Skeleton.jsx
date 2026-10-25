export default function Skeleton({ variant = "text", rows = 3, height }) {
  if (variant === "text") {
    return (
      <div className="skeleton-text">
        {Array.from({ length: rows }, (_, i) => (
          <div key={i} className="skeleton-line" style={{ width: `${100 - (i % 3) * 12}%` }} />
        ))}
      </div>
    );
  }
  const cls = variant === "chart" ? "skeleton skeleton-chart" : "skeleton skeleton-block";
  return <div className={cls} style={height ? { height } : undefined} />;
}
