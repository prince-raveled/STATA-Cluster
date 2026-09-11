# Export Pelto et al.'s curated cohorts to plain TSV for the Python harness.
#
# SPEC §23 validation 3 is "reproduce Pelto et al." — their curated data ships in the
# Zenodo archive (10.5281/zenodo.15047338) as `data_171023.rds`, which despite the
# extension is a `save()` image holding one plain list: each element is
# `list(meta = data.frame, counts = matrix[samples, taxa])`. No Bioconductor classes
# are involved, so this only needs base R.
#
#   Rscript pelto_export.R <data_171023.rds> <outdir>
#
# Writes <outdir>/index.csv plus <outdir>/<slug>/{counts.tsv,meta.tsv} per cohort.
# Counts are transposed to taxa x samples, which is what MicroVerse's parser expects.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: pelto_export.R data_171023.rds outdir")
rds_path <- args[1]
outdir <- args[2]

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

env <- new.env()
loaded <- load(rds_path, envir = env)
datasets <- get(loaded[1], envir = env)
writeLines(sprintf("loaded %d cohorts from %s", length(datasets), rds_path))

first <- function(x) if (length(x) == 0) NA else x[[1]]

slugify <- function(text) {
  text <- gsub("[^A-Za-z0-9]+", "_", text)
  text <- gsub("^_+|_+$", "", text)
  substr(text, 1, 80)
}

rows <- list()
for (i in seq_along(datasets)) {
  entry <- datasets[[i]]
  meta <- entry$meta
  counts <- entry$counts

  data_id <- first(meta$data_id)
  if (is.na(data_id)) data_id <- sprintf("cohort_%03d", i)
  slug <- slugify(data_id)
  iter <- first(meta$iter)
  half <- first(meta$half)
  kind <- if (is.na(iter)) "whole" else "half"

  cohort_dir <- file.path(outdir, slug)
  dir.create(cohort_dir, showWarnings = FALSE, recursive = TRUE)

  # taxa x samples, with an explicit id column: the shape the parser reads directly.
  wide <- t(counts)
  sample_ids <- colnames(wide)
  if (is.null(sample_ids)) sample_ids <- sprintf("S%04d", seq_len(ncol(wide)))
  colnames(wide) <- sample_ids
  frame <- data.frame(taxon = rownames(wide), wide, check.names = FALSE)
  write.table(frame, file.path(cohort_dir, "counts.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE)

  keep_cols <- intersect(c("group", "age", "sex", "bmi"), names(meta))
  sample_meta <- data.frame(sample_id = sample_ids, meta[, keep_cols, drop = FALSE],
                            check.names = FALSE)
  write.table(sample_meta, file.path(cohort_dir, "meta.tsv"),
              sep = "\t", quote = FALSE, row.names = FALSE, na = "")

  rows[[length(rows) + 1]] <- data.frame(
    slug = slug,
    data_id = data_id,
    kind = kind,
    type = first(meta$type),
    study = first(meta$study),
    disease = first(meta$disease),
    half = if (is.na(half)) NA_integer_ else as.integer(half),
    iter = if (is.na(iter)) NA_integer_ else as.integer(iter),
    n_samples = nrow(counts),
    n_taxa = ncol(counts),
    n_case = sum(meta$group == "case", na.rm = TRUE),
    n_control = sum(meta$group == "control", na.rm = TRUE),
    stringsAsFactors = FALSE
  )
}

index <- do.call(rbind, rows)
write.csv(index, file.path(outdir, "index.csv"), row.names = FALSE)
writeLines(sprintf("wrote %d cohorts (%d whole, %d split halves) to %s",
                   nrow(index), sum(index$kind == "whole"),
                   sum(index$kind == "half"), outdir))
writeLines(sprintf("R %s", getRversion()))
