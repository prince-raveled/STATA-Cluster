/* Result figures: the robustness landscape, the forest plot of findings that held up,
   and how labels move between weighting schemes. Each draws from the per-taxon records
   the page already holds, in the label colours the rest of the page uses.

   Taxon names come from an uploaded file, so they only ever reach Plotly as data
   (hover text, tick labels), never as markup. */
(function (global) {
  "use strict";

  const TIER_VAR = {
    "ROBUST": "--t-robust", "CONDITIONAL": "--t-conditional", "FRAGILE": "--t-fragile",
    "UNSTABLE": "--t-unstable", "INSUFFICIENT": "--t-insufficient",
    "NOT DETECTED": "--t-undetected"
  };
  const CLAIMS = ["ROBUST", "CONDITIONAL", "FRAGILE", "UNSTABLE"];

  function css(name, fallback) {
    const value = getComputedStyle(document.body).getPropertyValue(name);
    return (value || fallback).trim();
  }
  function tierColor(tier) { return css(TIER_VAR[tier] || "--ink-3", "#968385"); }
  const isNum = (v) => typeof v === "number" && isFinite(v);
  /* Plotly's responsive mode sizes a plot to its box, so the box gets the figure's own
     height first; otherwise a resize shrinks a tall figure to the box's minimum. */
  function sized(el, height) { el.style.height = height + "px"; }
  const pct = (v) => isNum(v) ? Math.round(v * 100) + "%" : "–";

  function baseLayout(t, extra) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: t.sans, color: t.ink, size: 12.5 },
      hoverlabel: { font: { family: t.sans, size: 12 } },
      showlegend: false
    }, extra);
  }
  const config = {
    displaylogo: false, responsive: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"],
    toImageButtonOptions: { format: "png", scale: 2 }
  };

  /* Figure 1 — every tested taxon at once: typical effect against how often significant. */
  function landscape(el, records, opts) {
    if (!el || !global.Plotly) return;
    const t = opts.theme;
    const traces = [];
    ["NOT DETECTED", "INSUFFICIENT", "UNSTABLE", "FRAGILE", "CONDITIONAL", "ROBUST"].forEach(function (tier) {
      const rows = records.filter((r) => r.tier === tier && isNum(r.median_effect));
      if (!rows.length) return;
      const quiet = tier === "NOT DETECTED" || tier === "INSUFFICIENT";
      traces.push({
        type: "scattergl", mode: "markers", name: tier,
        x: rows.map((r) => r.median_effect), y: rows.map((r) => r.frac_significant),
        customdata: rows.map((r) => [r.label, tier, pct(r.sign_consistency), r.direction,
                                     r.certified_robust ? " · CERTIFIED ROBUST" : "", r.taxon_id]),
        hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]}%{customdata[4]}<br>"
          + "significant in %{y:.0%} · direction agreement %{customdata[2]}<br>"
          + "typical effect %{x:.2f} (log2) · %{customdata[3]}<extra></extra>",
        marker: {
          color: tierColor(tier), size: quiet ? 5 : 9, opacity: quiet ? 0.45 : 0.9,
          symbol: rows.map((r) => r.certified_robust ? "star" : "circle"),
          line: { width: quiet ? 0 : 1, color: "rgba(255,255,255,0.9)" }
        }
      });
    });
    const shapes = [0.3, 0.8].map(function (y) {
      return { type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y,
               line: { color: t.line, width: 1.2, dash: "dot" } };
    });
    shapes.push({ type: "line", yref: "paper", x0: 0, x1: 0, y0: 0, y1: 1,
                  line: { color: t.line, width: 1 } });
    const annotations = [
      { xref: "paper", x: 1, y: 0.8, yanchor: "bottom", xanchor: "right", showarrow: false,
        text: "80% · ROBUST if direction agrees", font: { size: 11.5, color: t.inkSoft } },
      { xref: "paper", x: 1, y: 0.3, yanchor: "bottom", xanchor: "right", showarrow: false,
        text: "30% · CONDITIONAL", font: { size: 11.5, color: t.inkSoft } },
      { xref: "paper", yref: "paper", x: 0, y: 1.02, xanchor: "left", yanchor: "bottom",
        showarrow: false, text: "← higher in " + opts.groups.a, font: { size: 11.5, color: t.inkSoft } },
      { xref: "paper", yref: "paper", x: 1, y: 1.02, xanchor: "right", yanchor: "bottom",
        showarrow: false, text: "higher in " + opts.groups.b + " →", font: { size: 11.5, color: t.inkSoft } }
    ];
    sized(el, 380);
    global.Plotly.newPlot(el, traces, baseLayout(t, {
      height: 380, margin: { l: 56, r: 12, t: 26, b: 46 },
      xaxis: { title: { text: "typical effect (log2 fold change)", font: { size: 12 } },
               gridcolor: t.lineSoft, zeroline: false },
      yaxis: { title: { text: "share of analyses significant", font: { size: 12 } },
               tickformat: ".0%", range: [-0.03, 1.05], gridcolor: t.lineSoft, zeroline: false },
      shapes: shapes, annotations: annotations
    }), config);
    if (opts.onPick) el.on("plotly_click", function (event) {
      const point = event && event.points && event.points[0];
      if (point && point.customdata) opts.onPick(point.customdata[5]);
    });
  }

  /* Figure 2 — findings that held up: typical effect with its middle half, per taxon. */
  function forest(el, records, opts) {
    if (!el || !global.Plotly) return 0;
    const t = opts.theme;
    let rows = records.filter((r) => (r.tier === "ROBUST" || r.tier === "CONDITIONAL"
                                     || r.certified_robust) && isNum(r.median_effect));
    if (!rows.length) return 0;
    const limit = opts.limit || 30;
    if (rows.length > limit) {
      rows = rows.slice().sort((a, b) => b.frac_significant - a.frac_significant).slice(0, limit);
    }
    rows = rows.slice().sort((a, b) => a.median_effect - b.median_effect);
    const y = rows.map((r, i) => i);
    const ticks = rows.map((r) => r.label + "  " + pct(r.frac_significant));
    const height = Math.max(180, 70 + rows.length * 26);
    sized(el, height);
    global.Plotly.newPlot(el, [{
      type: "scatter", mode: "markers", x: rows.map((r) => r.median_effect), y: y,
      error_x: { type: "data", symmetric: false, thickness: 1.8, width: 0, color: t.inkSoft,
                 array: rows.map((r) => isNum(r.iqr_high) ? r.iqr_high - r.median_effect : 0),
                 arrayminus: rows.map((r) => isNum(r.iqr_low) ? r.median_effect - r.iqr_low : 0) },
      marker: { size: rows.map((r) => r.certified_robust ? 15 : 11),
                symbol: rows.map((r) => r.certified_robust ? "star" : "circle"),
                color: rows.map((r) => tierColor(r.tier)),
                line: { width: 1, color: "rgba(255,255,255,0.9)" } },
      customdata: rows.map((r) => [r.label, r.tier + (r.certified_robust ? " · CERTIFIED ROBUST" : ""),
                                   pct(r.frac_significant), pct(r.sign_consistency), r.taxon_id]),
      hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>significant in %{customdata[2]}"
        + " · direction agreement %{customdata[3]}<br>typical effect %{x:.2f}<extra></extra>"
    }], baseLayout(t, {
      height: height,
      margin: { l: 10, r: 16, t: 8, b: 44 },
      xaxis: { title: { text: "typical effect (log2 fold change), bar = middle half", font: { size: 12 } },
               gridcolor: t.lineSoft, zeroline: true, zerolinecolor: t.line, zerolinewidth: 1.5 },
      yaxis: { tickvals: y, ticktext: ticks, automargin: true, gridcolor: "rgba(0,0,0,0)",
               tickfont: { family: t.mono, size: 11.5, color: t.ink } }
    }), config);
    if (opts.onPick) el.on("plotly_click", function (event) {
      const point = event && event.points && event.points[0];
      if (point && point.customdata) opts.onPick(point.customdata[4]);
    });
    return rows.length;
  }

  /* Figure 3 — how each taxon's label moves from one vote per analysis to the weights the
     page uses. Taxa never detected under either scheme are left out of the flow. */
  function labelFlow(el, records, opts) {
    if (!el || !global.Plotly) return 0;
    const from = opts.from, to = opts.to;
    const rows = records.filter((r) => r.tiers && r.tiers[from] && r.tiers[to]
      && !(r.tiers[from] === "NOT DETECTED" && r.tiers[to] === "NOT DETECTED"));
    if (!rows.length) return 0;
    const order = CLAIMS.concat(["INSUFFICIENT", "NOT DETECTED"]);
    const used = (key) => order.filter((tier) => rows.some((r) => r.tiers[key] === tier));
    const left = used(from), right = used(to);
    const labels = left.concat(right);
    const counts = {};
    rows.forEach(function (r) {
      const k = r.tiers[from] + "|" + r.tiers[to];
      counts[k] = (counts[k] || 0) + 1;
    });
    const source = [], target = [], value = [], color = [];
    Object.keys(counts).forEach(function (k) {
      const parts = k.split("|");
      source.push(left.indexOf(parts[0]));
      target.push(left.length + right.indexOf(parts[1]));
      value.push(counts[k]);
      color.push(parts[0] === parts[1] ? "rgba(150,131,133,0.16)" : "rgba(212,99,161,0.32)");
    });
    const total = (key, tier) => rows.filter((r) => r.tiers[key] === tier).length;
    sized(el, 320);
    global.Plotly.newPlot(el, [{
      type: "sankey", arrangement: "fixed",
      textfont: { family: opts.theme.sans, size: 13, color: opts.theme.ink },
      node: { label: left.map((tier) => tier + " " + total(from, tier))
                       .concat(right.map((tier) => tier + " " + total(to, tier))),
              color: labels.map(tierColor), pad: 14, thickness: 14,
              line: { width: 0 }, hovertemplate: "%{label}<extra></extra>" },
      link: { source: source, target: target, value: value, color: color,
              hovertemplate: "%{source.label} → %{target.label}: %{value} taxa<extra></extra>" }
    }], baseLayout(opts.theme, {
      height: 320, margin: { l: 8, r: 8, t: 8, b: 8 }
    }), config);
    /* The two column titles sit above the plot, in the page, where Plotly's toolbar
       cannot cover them. */
    const ends = el.previousElementSibling;
    if (ends && ends.classList.contains("flow-ends")) {
      ends.children[0].textContent = opts.fromLabel;
      ends.children[1].textContent = opts.toLabel;
    }
    return rows.filter((r) => r.tiers[from] !== r.tiers[to]).length;
  }

  global.MicroVerseFigures = { landscape: landscape, forest: forest, labelFlow: labelFlow,
                               tierColor: tierColor };
})(window);
