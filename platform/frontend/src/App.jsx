import React, { useEffect, useState } from "react";
import { applyTheme, palette } from "./theme.js";
import MarketDashboard from "./pages/MarketDashboard.jsx";
import StrategyTrading from "./pages/StrategyTrading.jsx";

const NAV = [
  { key: "market", label: "市场看板" },
  { key: "strategy", label: "策略交易" },
];

export default function App() {
  const [page, setPage] = useState("market");
  // 展示习惯与主题由外壳持有，两页共享；环境各页自行管理
  const [convention, setConvention] = useState("greenUp");
  const [theme, setTheme] = useState("dark");
  const p = palette(convention, theme);

  useEffect(() => {
    applyTheme(document.documentElement, convention, theme);
  }, [convention, theme]);

  return (
    <div className="app">
      <nav className="topnav" aria-label="主导航">
        <div className="topnav-inner">
          <div className="nav-tabs">
            {NAV.map((n) => (
              <button
                key={n.key}
                type="button"
                className={`nav-tab ${page === n.key ? "active" : ""}`.trim()}
                aria-current={page === n.key ? "page" : undefined}
                onClick={() => setPage(n.key)}
              >
                {n.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* 两页常驻挂载，切换不丢失各自状态 */}
      <div className={page === "market" ? "" : "hidden"}>
        <MarketDashboard
          palette={p}
          convention={convention}
          theme={theme}
          onConventionChange={setConvention}
          onThemeChange={setTheme}
        />
      </div>
      <div className={page === "strategy" ? "" : "hidden"}>
        <StrategyTrading
          palette={p}
          convention={convention}
          theme={theme}
          onConventionChange={setConvention}
          onThemeChange={setTheme}
        />
      </div>
    </div>
  );
}
