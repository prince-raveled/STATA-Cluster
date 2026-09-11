# Export Tierney et al.'s cohorts from curatedMetagenomicData to plain TSV.
#
# SPEC §23 validation 1: "Run Mode C on their cohorts; check you recover 1-in-3
# sign-flipping and >90% T1D/T2D nonrobust." Tierney et al. 2022 (PLOS Biol
# 20(3):e3001556) drew their cohorts from curatedMetagenomicData and vibrated over
# covariate adjustment sets, which is exactly MicroVerse's covariate mode (§12).
#
#   Rscript tierney_export.R <outdir>
#
# Writes <outdir>/index.csv plus <outdir>/<slug>/{counts.tsv,meta.tsv} per contrast.
# Counts are species x samples with full MetaPhlAn lineages, so fork 4 has two levels
# (species and genus) rather than the one level a pre-collapsed table would give.

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "data/cmd/export"

lib <- Sys.getenv("MICROVERSE_R_LIB", unset = "D:/Rlocal/library")
if (dir.exists(lib)) .libPaths(lib)
options(timeout = 7200)

suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(dplyr)
})

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

#: The T1D and T2D contrasts §23.1 names, restricted to stool and to contrasts with
#: enough samples on both sides to fit an adjusted model at all.
CONTRASTS <- list(
  list(study = "Heitz-BuschartA_2016",   condition = "T1D"),
  list(study = "KosticAD_2015",          condition = "T1D"),
  list(study = "LiJ_2014",               condition = "T1D"),
  list(study = "HMP_2019_t2d",           condition = "T2D"),
  list(study = "KarlssonFH_2013",        condition = "T2D"),
  list(study = "MetaCardis_2020_a",      condition = "T2D"),
  list(study = "QinJ_2012",              condition = "T2D"),
  list(study = "SankaranarayananK_2015", condition = "T2D")
)

#: Tierney vibrated over adjustment sets built from the demographic and technical
#: variables cMD carries. `country` is constant inside a single study, so it is dropped
#: automatically by the completeness/variation filter below.
CANDIDATE_COVARIATES <- c("age", "gender", "BMI", "antibiotics_current_use",
                          "number_reads")

MIN_COMPLETE <- 0.90   # keep a covariate only if this fraction of samples have it
MIN_PER_GROUP <- 8

slugify <- function(text) {
  text <- gsub("[^A-Za-z0-9]+", "_", text)
  gsub("^_+|_+$", "", text)
}

rows <- list()
for (contrast in CONTRASTS) {
  study <- contrast$study
  condition <- contrast$condition
  slug <- slugify(paste(study, condition, sep = "_"))
  writeLines(sprintf("--- %s (%s)", study, condition))

  meta <- sampleMetadata |>
    filter(study_name == study, body_site == "stool",
           study_condition %in% c(condition, "control"))
  if (nrow(meta) == 0) {
    writeLines("    no samples; skipped")
    next
  }

  # Keep only covariates that are well populated *and* actually vary here: a constant
  # column adds a fork level that cannot change any answer.
  keep <- c()
  for (name in CANDIDATE_COVARIATES) {
    if (!name %in% names(meta)) next
    values <- meta[[name]]
    if (mean(!is.na(values)) < MIN_COMPLETE) next
    if (length(unique(values[!is.na(values)])) < 2) next
    keep <- c(keep, name)
  }
  if (length(keep) > 0) {
    complete <- stats::complete.cases(meta[, keep, drop = FALSE])
    meta <- meta[complete, ]
  }

  n_case <- sum(meta$study_condition == condition)
  n_control <- sum(meta$study_condition == "control")
  if (n_case < MIN_PER_GROUP || n_control < MIN_PER_GROUP) {
    writeLines(sprintf("    %d case / %d control after dropping incomplete rows; skipped",
                       n_case, n_control))
    next
  }

  tse <- returnSamples(meta, "relative_abundance", counts = TRUE, rownames = "long")
  counts <- SummarizedExperiment::assay(tse)
  storage.mode(counts) <- "integer"

  cohort_dir <- file.path(outdir, slug)
  dir.create(cohort_dir, showWarnings = FALSE, recursive = TRUE)

  frame <- data.frame(taxon = rownames(counts), counts, check.names = FALSE)
  write.table(frame, file.path(cohort_dir, "counts.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)

  ordered <- meta[match(colnames(counts), meta$sample_id), ]
  sample_meta <- data.frame(
    sample_id = colnames(counts),
    group = ifelse(ordered$study_condition == condition, "b_case", "a_control"),
    check.names = FALSE
  )
  for (name in keep) sample_meta[[name]] <- ordered[[name]]
  write.table(sample_meta, file.path(cohort_dir, "meta.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE, na = "")

  writeLines(sprintf("    %d taxa x %d samples; case=%d control=%d; covariates: %s",
                     nrow(counts), ncol(counts), n_case, n_control,
                     paste(keep, collapse = ", ")))

  rows[[length(rows) + 1]] <- data.frame(
    slug = slug, study = study, condition = condition,
    n_samples = ncol(counts), n_taxa = nrow(counts),
    n_case = n_case, n_control = n_control,
    covariates = paste(keep, collapse = "|"),
    stringsAsFactors = FALSE
  )
}

index <- do.call(rbind, rows)
write.csv(index, file.path(outdir, "index.csv"), row.names = FALSE)
writeLines(sprintf("wrote %d contrasts to %s", nrow(index), outdir))
writeLines(sprintf("curatedMetagenomicData %s | R %s",
                   packageVersion("curatedMetagenomicData"), getRversion()))
