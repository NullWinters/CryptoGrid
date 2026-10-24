// 展示习惯 convention: "greenUp" 绿涨红跌 | "redUp" 红涨绿跌
// 主题 theme: "light" | "dark"
const GREEN = "#16a34a";
const RED = "#dc2626";

const THEMES = {
  light: { bg: "#ffffff", fg: "#111827", border: "#d1d5db", muted: "#6b7280" },
  dark: { bg: "#111827", fg: "#e5e7eb", border: "#374151", muted: "#9ca3af" },
};

export function palette(convention, theme) {
  const greenUp = convention !== "redUp";
  return {
    up: greenUp ? GREEN : RED,
    down: greenUp ? RED : GREEN,
    ...(THEMES[theme] ?? THEMES.light),
  };
}
