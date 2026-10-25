// 设计令牌与主题应用。颜色只在此处定义，DOM 与图表共用同一份来源。
// 展示习惯 convention: "greenUp" 绿涨红跌 | "redUp" 红涨绿跌
// 主题 theme: "dark" | "light"

export const TOKENS = {
  dark: {
    bg: "#0B0E14",
    surface: "#131722",
    surface2: "#1A1F2B",
    border: "#2A2E39",
    borderStrong: "#636B80",
    fg: "#D1D4DC",
    muted: "#8A8F9C",
    accent: "#2962FF",
    accentText: "#6E9BFF",
    accentSoft: "rgba(41,98,255,0.12)",
    up: "#26A69A",
    down: "#EF5350",
    upText: "#26A69A",
    downText: "#EF5350",
    warn: "#F5A623",
    danger: "#EF5350",
    dangerSolid: "#C62828",
    focus: "#2962FF",
  },
  light: {
    bg: "#FFFFFF",
    surface: "#F8F9FB",
    surface2: "#F0F2F5",
    border: "#E0E3EB",
    borderStrong: "#8A8F9C",
    fg: "#131722",
    muted: "#5D606B",
    accent: "#2962FF",
    accentText: "#1B4FD8",
    accentSoft: "rgba(41,98,255,0.08)",
    up: "#089981",
    down: "#F23645",
    upText: "#007A69",
    downText: "#C62828",
    warn: "#8A5A00",
    danger: "#D93025",
    dangerSolid: "#D93025",
    focus: "#2962FF",
  },
};

const CSS_VARS = {
  bg: "--bg",
  surface: "--surface",
  surface2: "--surface-2",
  border: "--border",
  borderStrong: "--border-strong",
  fg: "--fg",
  muted: "--fg-muted",
  accent: "--accent",
  accentText: "--accent-text",
  accentSoft: "--accent-soft",
  up: "--up",
  down: "--down",
  upText: "--up-text",
  downText: "--down-text",
  warn: "--warn",
  danger: "--danger",
  dangerSolid: "--danger-solid",
  focus: "--focus",
};

export function palette(convention, theme) {
  const base = TOKENS[theme] ?? TOKENS.dark;
  const greenUp = convention !== "redUp";
  return {
    ...base,
    up: greenUp ? base.up : base.down,
    down: greenUp ? base.down : base.up,
    upText: greenUp ? base.upText : base.downText,
    downText: greenUp ? base.downText : base.upText,
  };
}

export function applyTheme(root, convention, theme) {
  const tokens = palette(convention, theme);
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
  for (const [key, cssVar] of Object.entries(CSS_VARS)) {
    root.style.setProperty(cssVar, tokens[key]);
  }
}
