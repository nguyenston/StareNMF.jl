setwd("../WGS_PCAWG.96.ready/")
library(musicatk)
library(readr)

.extract_count_table <- function(musica, table_name) {
  # Check that at least one table exists
  if (length(tables(musica)) == 0) {
    stop(strwrap(prefix = " ", initial = "", "The counts table is either
    missing or malformed, please run create tables e.g. [build_standard_table]
    prior to this function."))
  }

  # Check that table exists within this musica
  if (!table_name %in% names(tables(musica))) {
    stop(paste0(
      "The table '", table_name, "' does not exist. ",
      "Tables in the 'musica' object include: ",
      paste(names(tables(musica)), collapse = ", ")
    ))
  }

  return(extract_count_tables(musica)[[table_name]]@count_table)
}

discover_signatures <- function(musica, table_name, num_signatures,
                                algorithm = "lda", seed = 1, nstart = 10,
                                par_cores = 1) {
  if (!methods::is(musica, "musica")) {
    stop("Input to discover_signatures must be a 'musica' object.")
  }
  algorithm <- match.arg(algorithm, c("lda", "nmf"))
  counts_table <- .extract_count_table(musica, table_name)
  if (dim(counts_table)[2] < 2) {
    stop("The 'musica' object inputted must contain at least two samples.")
  }
  present_samples <- which(colSums(counts_table) > 0)
  counts_table <- counts_table[, present_samples]

  if (algorithm == "lda") {
    lda_counts_table <- t(counts_table)
    if (is.null(seed)) {
      control <- list(nstart = nstart)
    } else {
      control <- list(seed = (seq_len(nstart) - 1) + seed, nstart = nstart)
    }

    lda_out <- topicmodels::LDA(lda_counts_table, num_signatures,
      control = control
    )

    lda_sigs <- exp(t(lda_out@beta))
    rownames(lda_sigs) <- colnames(lda_counts_table)
    colnames(lda_sigs) <- paste0("Signature", seq_len(num_signatures))

    weights <- t(lda_out@gamma)
    rownames(weights) <- paste0("Signature", seq_len(num_signatures))
    colnames(weights) <- rownames(lda_counts_table)

    result <- methods::new("musica_result",
      signatures = lda_sigs,
      table_name = table_name,
      exposures = weights, algorithm = "LDA",
      musica = musica
    )
    exposures(result) <- sweep(exposures(result), 2, colSums(exposures(result)),
      FUN = "/"
    )
  } else if (algorithm == "nmf") {
    # Needed to prevent error with entirely zero rows
    epsilon <- 0.00000001

    decomp <- NMF::nmf(counts_table + epsilon, num_signatures, "brunet",
      seed = seed,
      nrun = nstart, .options = paste("p", par_cores,
        sep = ""
      )
    )

    rownames(decomp@fit@H) <- paste("Signature", seq_len(num_signatures),
      sep = ""
    )
    colnames(decomp@fit@W) <- paste("Signature", seq_len(num_signatures),
      sep = ""
    )
    result <- methods::new("musica_result",
      signatures = decomp@fit@W,
      table_name = table_name,
      exposures = decomp@fit@H, algorithm = "NMF",
      musica = musica
    )
    signatures(result) <- sweep(signatures(result), 2,
      colSums(signatures(result)),
      FUN = "/"
    )
    exposures(result) <- sweep(exposures(result), 2, colSums(exposures(result)),
      FUN = "/"
    )
  } else {
    stop(
      "That method is not supported. Please select 'lda' or 'nmf' ",
      "to generate signatures."
    )
  }
  # Multiply Weights by sample counts
  sample_counts <- colSums(counts_table)
  matched <- match(colnames(counts_table), names(sample_counts))
  exposures(result) <- sweep(exposures(result), 2, sample_counts[matched],
    FUN = "*"
  )
  return(result)
}

cancer_categories <- list(
  skin = "107-skin-melanoma-all-seed-1",
  ovary = "113-ovary-adenoca-all-seed-1",
  breast = "214-breast-all-seed-1",
  liver = "326-liver-hcc-all-seed-1",
  lung = "38-lung-adenoca-all-seed-1",
  stomach = "75-stomach-adenoca-all-seed-1"
)
misspecification_type <- list(
  none = "",
  contaminated = "-contamination-2",
  overdispersed = "-overdispersed-2.0",
  perturbed = "-perturbed-0.0025"
)
k_min <- 1
k_max <- 21

# override current musicatk function for creating
#   a musica object to work with only a count table
create_musica <- function(count_table) {
  # create empty musica object
  musica <- new("musica")

  # create SBS mutation type list
  forward_change <- c("C>A", "C>G", "C>T", "T>A", "T>C", "T>G")
  b1 <- rep(rep(c("A", "C", "G", "T"), each = 4), 6)
  b2 <- rep(c("C", "T"), each = 48)
  b3 <- rep(c("A", "C", "G", "T"), 24)
  mut_trinuc <- apply(cbind(b1, b2, b3), 1, paste, collapse = "")
  mut <- rep(forward_change, each = 16)
  annotation <- data.frame(
    "motif" = paste0(mut, "_", mut_trinuc),
    "mutation" = mut,
    "context" = mut_trinuc
  )
  rownames(annotation) <- annotation$motif

  # color mapping for mutation types
  color_mapping <- c(
    "C>A" = "#5ABCEBFF",
    "C>G" = "#050708FF",
    "C>T" = "#D33C32FF",
    "T>A" = "#CBCACBFF",
    "T>C" = "#ABCD72FF",
    "T>G" = "#E7C9C6FF"
  )

  # update count table rownames with SBS96 standard naming
  rownames(count_table) <- annotation$motif

  # create count table object
  tab <- new("count_table",
    name = "SBS96", count_table = as.matrix(count_table),
    annotation = annotation, features = as.data.frame(annotation$motif[1]),
    type = S4Vectors::Rle("SBS"), color_variable = "mutation",
    color_mapping = color_mapping, description = paste0(
      "Single Base Substitution table with",
      " one base upstream and downstream"
    )
  )

  # add count table to musica object
  tables(musica)[["SBS96"]] <- tab

  return(musica)
}

# loop through each dataset
for (cancer in names(cancer_categories)) {
  for (misspec in names(misspecification_type)) {
    file <- paste(
      "synthetic-", cancer_categories$cancer,
      misspecification_type$misspec, ".tsv",
      sep = ""
    )
    # get tumor type
    tumor_type <- cancer

    # read in count table
    count_table <- as.data.frame(read_tsv(file))
    # remove column of mutation types
    count_table <- count_table[, c(-1)]
    # specify rownames as the mutation types (use format used by cosmic)
    rownames(count_table) <- rownames(signatures(cosmic_v3_sbs_sigs))

    # create musica object with mutation counts
    musica <- create_musica(count_table)

    # perform signature discovery several times with diff k values
    for (k in k_min:k_max) {
      file_name <- paste(cancer, "-", misspec, k, sep = "")
      result <- discover_signatures(musica, "SBS96", k, "nmf")
      write.csv(signatures(result), paste(
        "../raw-cache-R/synthetic/nmf/",
        file_name, "-W.csv",
        sep = ""
      ))
      write.csv(exposures(result), paste(
        "../raw-cache-R/synthetic/nmf/", "-H.csv",
        sep = ""
      ))
    }
  }
}
