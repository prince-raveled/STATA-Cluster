# Reference implementations, run in R, for the Python harness to diff against.
#
# SPEC §10 names edgeR's TMM, ALDEx2 and ANCOM-BC, all three of which are R-only.
# Everything the in-tree code claims to reproduce is computed here by the actual
# packages and written to CSV; `tests/reference/compare_r.py` reads the results back
# and compares.
#
#   Rscript r_reference.R <counts.csv> <groups.csv> <outdir> [aldex_mc_samples]
#
# counts.csv : taxa x samples, first column the taxon id, header the sample ids
# groups.csv : sample_id,group   with exactly two group levels

suppressPackageStartupMessages({
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 3) stop("usage: r_reference.R counts.csv groups.csv outdir [mc.samples]")
  counts_path <- args[1]
  groups_path <- args[2]
  outdir      <- args[3]
  mc_samples  <- if (length(args) >= 4) as.integer(args[4]) else 128L

  lib <- Sys.getenv("MICROVERSE_R_LIB", unset = "D:/Rlocal/library")
  if (dir.exists(lib)) .libPaths(lib)
  library(edgeR)
  library(ALDEx2)
  # ancombc() routes its input through the microbiome package, so all three have to
  # be present before the comparison is worth attempting.
  have_ancombc <- all(vapply(c("ANCOMBC", "TreeSummarizedExperiment", "microbiome"),
                             requireNamespace, logical(1), quietly = TRUE))
  if (have_ancombc) {
    library(ANCOMBC)
    library(TreeSummarizedExperiment)
  }
})

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

counts <- as.matrix(read.csv(counts_path, row.names = 1, check.names = FALSE))
groups <- read.csv(groups_path, stringsAsFactors = FALSE)
stopifnot(all(colnames(counts) == groups$sample_id))
condition <- factor(groups$group)
stopifnot(nlevels(condition) == 2)

cat(sprintf("R %s | edgeR %s | ALDEx2 %s | ANCOMBC %s\n",
            getRversion(), packageVersion("edgeR"), packageVersion("ALDEx2"),
            if (have_ancombc) as.character(packageVersion("ANCOMBC")) else "absent"))
cat(sprintf("counts: %d taxa x %d samples; groups: %s\n",
            nrow(counts), ncol(counts), paste(levels(condition), collapse = " vs ")))

# ---------------------------------------------------------------------------
# edgeR — TMM normalisation factors (SPEC §10 fork 3)
# ---------------------------------------------------------------------------
# calcNormFactors returns factors centred to a geometric mean of 1, exactly what
# app/core/preprocess.py::tmm_factors returns.
dge <- DGEList(counts = counts, group = condition)
dge <- calcNormFactors(dge, method = "TMM")
write.csv(
  data.frame(sample = colnames(counts),
             lib_size = as.numeric(dge$samples$lib.size),
             norm_factor = as.numeric(dge$samples$norm.factors)),
  file.path(outdir, "edger_tmm.csv"), row.names = FALSE
)
cat("wrote edger_tmm.csv\n")

# ---------------------------------------------------------------------------
# ALDEx2 (SPEC §10 fork 5, method 6)
# ---------------------------------------------------------------------------
# aldex() runs the Dirichlet Monte-Carlo, CLR-transforms each instance (in log2) and
# returns the expected p-values (wi.ep = Wilcoxon, we.ep = Welch) plus diff.btw, the
# median of randomly paired between-group differences in CLR values.
set.seed(1)
ax <- aldex(reads = as.data.frame(counts), conditions = as.character(condition),
            mc.samples = mc_samples, test = "t", effect = TRUE, denom = "all",
            verbose = FALSE)
write.csv(
  data.frame(taxon = rownames(ax),
             diff_btw = ax$diff.btw,      # median CLR difference between groups
             effect = ax$effect,
             we_ep = ax$we.ep,            # expected Welch p-value
             wi_ep = ax$wi.ep,            # expected Wilcoxon p-value
             we_eBH = ax$we.eBH,
             wi_eBH = ax$wi.eBH),
  file.path(outdir, "aldex2.csv"), row.names = FALSE
)
cat(sprintf("wrote aldex2.csv (mc.samples = %d)\n", mc_samples))

# ---------------------------------------------------------------------------
# ANCOM-BC (SPEC §10 fork 5, method 5)
# ---------------------------------------------------------------------------
# The v1 `ancombc()` is the Lin & Peddada 2020 estimator that app/core/methods/
# ancombc.py implements: bias-corrected log abundances, sampling fractions from a
# normal-mixture E-M, then a Wald test per taxon. Filters are turned off so the
# comparison is of the estimator alone, not of ANCOM-BC's preprocessing.
if (have_ancombc) {
  meta <- data.frame(group = condition, row.names = colnames(counts))
  tse <- TreeSummarizedExperiment(assays = list(counts = counts), colData = meta)
  # A failure here must not cost the edgeR and ALDEx2 results already written.
  ok <- tryCatch({
    fit <- ancombc(data = tse, assay_name = "counts", formula = "group",
                   p_adj_method = "BH", prv_cut = 0, lib_cut = 0, group = "group",
                   struc_zero = FALSE, neg_lb = FALSE, tol = 1e-5, max_iter = 100,
                   conserve = FALSE, alpha = 0.05, global = FALSE)
    res <- fit$res
    term <- setdiff(colnames(res$lfc), c("taxon", "(Intercept)"))[1]
    write.csv(
      data.frame(taxon = res$lfc$taxon,
                 lfc = res$lfc[[term]],        # natural-log fold change
                 se = res$se[[term]],
                 W = res$W[[term]],
                 p_val = res$p_val[[term]],
                 q_val = res$q_val[[term]]),
      file.path(outdir, "ancombc.csv"), row.names = FALSE
    )
    cat(sprintf("wrote ancombc.csv (term = %s)\n", term))
    TRUE
  }, error = function(e) {
    cat(sprintf("ANCOM-BC failed, skipping ancombc.csv: %s\n", conditionMessage(e)))
    FALSE
  })
  have_ancombc <- ok
} else {
  cat("ANCOMBC (or microbiome / TreeSummarizedExperiment) not installed; skipped\n")
}

writeLines(c(
  paste0("r_version=", getRversion()),
  paste0("edger_version=", packageVersion("edgeR")),
  paste0("aldex2_version=", packageVersion("ALDEx2")),
  paste0("ancombc_version=",
         if (have_ancombc) as.character(packageVersion("ANCOMBC")) else "absent"),
  paste0("mc_samples=", mc_samples)
), file.path(outdir, "versions.txt"))
cat("done\n")
