/* Specification curve — SPEC §16.3.
 *
 * Simonsohn's two-panel plot. Upper: the harmonised effect for every tested
 * specification, sorted ascending, coloured by significance, zero reference line.
 * Lower: a dot matrix of which fork level was active in each specification,
 * x-aligned to the panel above.
 *
 * Colours come from the stylesheet so the plot and the page cannot drift apart.
 */
(function (global) {
  "use strict";

  function theme() {
    const style = getComputedStyle(document.body);
    const pick = (name, fallback) => (style.getPropertyValue(name) || fallback).trim();
    return {
      ink: pick("--ink", "#554345"),
      inkSoft: pick("--ink-2", "#6b585a"),
      grey: pick("--ink-3", "#735f61"),
      line: pick("--line", "#eddccb"),
      lineSoft: pick("--line-soft", "#f3e8dd"),
      /* Significance is the headline result, so it takes the page's one accent; the
         declared pipeline is a reference mark, in cornflower. */
      significant: pick("--accent", "#D463A1"),
      nonsignificant: "#9a8b8c",
      matrix: pick("--ink-3", "#735f61"),
      declared: pick("--focus", "#3f87be"),
      sans: pick("--sans", "sans-serif"),
      mono: pick("--mono", "monospace")
    };
  }

  /* Row prefixes on a narrow screen, where the fork names have no margin to sit in. */
  const SHORT_FORK = {
    rarefaction: "depth", rank: "rank", prev_filter: "prev", transform: "scale",
    method: "test", fdr_method: "FDR", fdr_threshold: "q ≤"
  };

  /* Flatten the fork -> categories map into stacked dot-matrix rows, with a
     spacer row between forks. */
  function buildRows(data) {
    const rows = [];
    Object.keys(data.forks).forEach(function (fork) {
      const categories = data.categories[fork] || [];
      if (categories.length < 2) return;  // a fork with one level shows nothing
      categories.forEach(function (level) {
        rows.push({ fork: fork, level: level, label: level });
      });
      rows.push({ fork: fork, level: null, label: "" });
    });
    return rows;
  }

  function render(elementId, data) {
    const colors = theme();
    const element = document.getElementById(elementId);
    if (!element) return;

    if (!data.n_specs) {
      element.innerHTML = '<p class="plot-empty">'
        + 'This taxon was not tested in any specification, so there is nothing to plot.</p>';
      return;
    }
    /* Narrow screens get a narrower label column rather than a squashed plot: the
       fork names move off the margin and each level carries its own label. */
    const narrow = element.clientWidth < 640;

    const n = data.effect.length;

    /* ---- upper panel: effect size, coloured by significance ---- */
    const sigX = [], sigY = [], sigText = [];
    const nullX = [], nullY = [], nullText = [];
    for (let i = 0; i < n; i++) {
      const label = data.labels[i]
        + "<br>effect " + data.effect[i].toFixed(3) + " log2FC"
        + "<br>adjusted p " + data.p_adjusted[i].toExponential(2);
      if (data.significant[i]) {
        sigX.push(i); sigY.push(data.effect[i]); sigText.push(label);
      } else {
        nullX.push(i); nullY.push(data.effect[i]); nullText.push(label);
      }
    }

    const traces = [
      {
        x: nullX, y: nullY, text: nullText, type: "scattergl", mode: "markers",
        name: "not significant", hovertemplate: "%{text}<extra></extra>",
        marker: { size: 4, color: colors.nonsignificant, opacity: 0.8 }, xaxis: "x", yaxis: "y"
      },
      {
        x: sigX, y: sigY, text: sigText, type: "scattergl", mode: "markers",
        name: "FDR significant", hovertemplate: "%{text}<extra></extra>",
        marker: { size: 5, color: colors.significant, opacity: 0.92 }, xaxis: "x", yaxis: "y"
      }
    ];

    /* ---- lower panel: the fork dot matrix ---- */
    const rows = buildRows(data);
    const rowIndex = new Map();
    rows.forEach(function (row, i) {
      if (row.level !== null) rowIndex.set(row.fork + "\u0000" + row.level, i);
    });

    const dotX = [], dotY = [], dotText = [];
    Object.keys(data.forks).forEach(function (fork) {
      const values = data.forks[fork];
      const forkLabel = data.fork_labels[fork] || fork;
      for (let i = 0; i < values.length; i++) {
        const key = fork + "\u0000" + values[i];
        if (!rowIndex.has(key)) continue;
        dotX.push(i);
        dotY.push(rowIndex.get(key));
        dotText.push(forkLabel + ": " + values[i]);
      }
    });

    traces.push({
      x: dotX, y: dotY, text: dotText, type: "scattergl", mode: "markers",
      name: "fork level", showlegend: false,
      hovertemplate: "%{text}<extra></extra>",
      marker: { size: 3, color: colors.matrix, opacity: 0.7 },
      xaxis: "x", yaxis: "y2"
    });

    /* ---- fork group labels in the margin, separators between blocks ----
     * The labels are shifted clear of the level ticks by a fixed pixel offset;
     * anchoring them to the plot edge alone puts them on top of the ticks.
     */
    const LABEL_SHIFT = -134;
    const shapes = [{
      type: "line", xref: "x", yref: "y", x0: -0.5, x1: n - 0.5, y0: 0, y1: 0,
      line: { color: colors.inkSoft, width: 1, dash: "dot" }
    }];
    const annotations = [];
    const seen = new Set();
    rows.forEach(function (row, i) {
      if (row.level === null) {
        if (i < rows.length - 1) {
          shapes.push({
            type: "line", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: i, y1: i,
            line: { color: colors.lineSoft, width: 1 }
          });
        }
        return;
      }
      if (narrow || seen.has(row.fork)) return;
      seen.add(row.fork);
      const size = (data.categories[row.fork] || []).length;
      annotations.push({
        xref: "paper", yref: "y2", x: 0, y: i + (size - 1) / 2,
        xanchor: "right", xshift: LABEL_SHIFT,
        text: data.fork_labels[row.fork] || row.fork,
        showarrow: false,
        font: { size: 10, color: colors.ink, family: colors.sans }
      });
    });

    if (data.declared_position >= 0) {
      ["y", "y2"].forEach(function (axis) {
        shapes.push({
          type: "line", xref: "x", yref: axis + " domain",
          x0: data.declared_position, x1: data.declared_position, y0: 0, y1: 1,
          line: { color: colors.declared, width: 1.5 }
        });
      });
      annotations.push({
        xref: "x", yref: "y domain", x: data.declared_position, y: 1.02,
        text: "your pipeline", showarrow: false, xanchor: "center",
        font: { size: 10, color: colors.declared, family: colors.mono }
      });
    }

    const tickvals = [];
    const ticktext = [];
    rows.forEach(function (row, i) {
      if (row.level === null) return;
      tickvals.push(i);
      ticktext.push(narrow ? (SHORT_FORK[row.fork] || row.fork) + ": " + row.label : row.label);
    });

    const layout = {
      height: Math.max(460, 320 + rows.length * 19),
      margin: { l: narrow ? Math.round(Math.min(150, Math.max(92, element.clientWidth * 0.34))) : 272,
                r: narrow ? 8 : 18, t: 38, b: 46 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: colors.sans, color: colors.grey, size: 11.5 },
      showlegend: true,
      legend: { orientation: "h", y: 1.09, x: 0, font: { size: 11 } },
      hovermode: "closest",
      hoverlabel: { font: { family: colors.sans, size: 11 } },
      grid: { rows: 2, columns: 1, pattern: "independent" },
      xaxis: {
        domain: [0, 1], anchor: "y2",
        title: { text: "specifications, sorted by effect size", font: { size: 11 } },
        showgrid: false, zeroline: false, color: colors.grey, linecolor: colors.line
      },
      yaxis: {
        domain: [0.58, 1], anchor: "x",
        title: {
          text: "log2 fold change (" + data.group_b + " vs " + data.group_a + ")",
          standoff: 6,
          font: { size: 11.5 }
        },
        gridcolor: colors.lineSoft, zerolinecolor: colors.inkSoft, color: colors.grey
      },
      yaxis2: {
        domain: [0, 0.52], anchor: "x", autorange: "reversed",
        tickmode: "array", tickvals: tickvals, ticktext: ticktext,
        tickfont: { size: element.clientWidth < 420 ? 8 : narrow ? 9 : 10, family: colors.mono },
        showgrid: false, zeroline: false, color: colors.grey
      },
      shapes: shapes,
      annotations: annotations
    };

    Plotly.newPlot(element, traces, layout, {
      displaylogo: false, responsive: true,
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d"],
      toImageButtonOptions: {
        filename: "specification_curve_" + data.taxon, scale: 2, format: "png"
      }
    });
  }

  global.MicroVerseCurve = { render: render, theme: theme };
})(window);
