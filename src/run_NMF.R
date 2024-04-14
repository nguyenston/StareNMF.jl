setwd("../WGS_PCAWG.96.ready/")
library(musicatk)
library(readr)


files <- c(
  "Breast.tsv", "Liver-HCC.tsv", "Lung-SCC.tsv", "Ovary-AdenoCA.tsv",
  "Skin-Melanoma.tsv", "Stomach-AdenoCA.tsv"
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
for (file in files) {
  # get tumor type
  tumor_type <- substr(file, 1, nchar(file) - 4)

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
    file_name <- paste(tumor_type, k, sep = "")
    result <- discover_signatures(musica, "SBS96", k, "nmf", par_cores = 16)
    write.csv(signatures(result), paste(file_name, "-W.csv", sep = ""))
    write.csv(exposures(result), paste(file_name, "-H.csv", sep = ""))
  }
}
