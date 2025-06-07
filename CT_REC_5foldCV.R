###CausalTree Model to Make Recommendation

install_r_packages <- function (){
  # Install R package from CRAN repository
  Rpackage = c("remotes", "BiocManager", "librarian")
  package.check <- lapply(
    Rpackage,
    FUN = function(x) {
      if (!require(x, character.only = TRUE)) {
        install.packages(x, repos='https://cloud.r-project.org', dependencies = TRUE)
        library(x, character.only = TRUE)
      }
    }
  )

  librarian::shelf(dplyr, RcppArmadillo, Matrix, arules, irlba,
                   igraph, robustbase, fastICA,  recommenderlab, grf, rpart.plot,
                   devtools, quiet = TRUE)

  BiocManager::install("Biobase")
  librarian::shelf(graph, RBGL, Rgraphviz)
  librarian::shelf(ggm,  pcalg )

}

# #Load R library
r_package = c("dplyr", "recommenderlab", "pcalg", "causalTree", "graph","grf","rpart.plot")
## Now load or install&load all
package.check <- lapply(
  r_package,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install_r_packages()
      library(x, character.only = TRUE)
    }
  }
)


# Install R package from Github repository
Rpackage5 = c("causalTree")
package.check <- lapply(
  Rpackage5,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      library(devtools)
      install_github("susanathey/causalTree")
      library(x, character.only = TRUE)
    }
  }
)

library(dplyr)
library(causalTree)
library(rpart.plot)

## # build Causal Tree model
build_causal_tree_model <- function(data_name, trainingData, causal_factors, outcome_name, splitRule = 'CT', cvRule = 'CT') {
  results <- list()
  output_folder = getwd()
  out_put <- file.path(output_folder, 'output')

  if (!dir.exists(out_put)) {
    dir.create(out_put)
  }
  outputDir <- file.path(out_put, 'trained_CTmodel')

  if (!dir.exists(outputDir)) {
    dir.create(outputDir)
  }

  for (i in 1:length(causal_factors)) {
    reg <- glm(as.formula(paste(causal_factors[[i]], ' ~ . -', outcome_name, sep = "")),
               family = binomial,
               data = trainingData)

    propensity_scores <- reg$fitted
    tree <- causalTree(as.formula(paste(outcome_name, ' ~ . ', sep = "")),
                       data = trainingData,
                       treatment = trainingData[[causal_factors[[i]]]],
                       split.Rule = splitRule,
                       cv.option = cvRule,
                       split.Honest = TRUE,
                       cv.Honest = TRUE,
                       split.Bucket = FALSE,
                       xval = 5,
                       cp = 0,
                       minsize = 3L,
                       propensity = propensity_scores)


    opcp <- tree$cptable[, 2][which.min(tree$cptable[, 4])]
    opfit <- prune(tree, opcp)

    treeFileName <- paste(data_name,'_', causal_factors[[i]], '_tree.png', sep = '')
    treeFile <- file.path(outputDir, treeFileName)
    png(file = treeFile, width = 1200, height = 900)

    rpart.plot(opfit)
    dev.off()

    treeModel <- list()
    treeModel$model <- opfit
    treeModel$factor <- causal_factors[[i]]
    results <- append(results, list(treeModel))

    trainingModel <- opfit
    trainedModelFileName <- paste(data_name, '_', causal_factors[[i]], '_trainedCTmodel.RDS', sep = '')

    trainedModelFile <- paste (outputDir, '/', trainedModelFileName,sep='')
    saveRDS(trainingModel, file=trainedModelFile)
  }
  return(results)
}

make_binary_recommendation_r <- function(method,
                                         trainedModel,
                                         input_data_folder,
                                         outputBaseFolder,
                                         data_name,
                                         outcome_name,
                                         remove_attributes = NULL,
                                         fold) {
  #Using the trained model from .RDS file or object from train_model() function.
  is_file <- function(obj) {
    if (is.character(obj) && file.exists(obj)) {
      return(TRUE)
    } else {
      return(FALSE)
    }
  }
  if (is_file(trainedModel)){     #check if trainedModel is file or object
    trainedModel <- readRDS(trainedModel)   # if trainedModel is a RDS file, read file.
  }
  else{
    trainedModel <- trainedModel
  }

  infileName <- paste (input_data_folder, '/', data_name, '_test.csv', sep='')
  data <-read.csv(file = infileName)
  data['HIGHEST_TE'] <- 0
  data['REC_TP'] <- ''
  data['CURRENT_TP'] <- ''
  data ['FOLLOW_REC'] <- 0



  for(row in 1: nrow(data)){
    for(i in 1: length(causal_factors)){
      if (data[row,causal_factors[[i]]] == 1){
        data[row,'CURRENT_TP'] <- causal_factors[[i]]
      }
    }
    inputRow = dplyr::select (data[row, ], -c('HIGHEST_TE', 'REC_TP','CURRENT_TP', 'FOLLOW_REC', outcome_name, remove_attributes ))
    # inputRow = dplyr::select (data[row, ], -c('HIGHEST_TE', 'REC_TP','CURRENT_TP', 'FOLLOW_REC', outcome_name))

    # val <- predict_causal_effect(trainedModel, inputRow, threshold)
    rowTE <- predict_causal_effect_row(trainedModel, inputRow, threshold)

    prevLift = -9999
    highestTreatmentName <- ''

    for(i in 1: length(causal_factors)){
      data[row,causal_factors[[i]]] <- rowTE[causal_factors[[i]]]
      # row[causal_factors[[i]]]

      if(prevLift < rowTE[causal_factors[[i]]]){
        prevLift <- rowTE[causal_factors[[i]]]
        if(prevLift > threshold) {
          highestTreatmentName <- causal_factors[[i]]
        }
      }
    }
    data[row,'HIGHEST_TE'] <- prevLift
    data[row,'REC_TP'] <- highestTreatmentName
    if((data[row,'REC_TP'] == data[row,'CURRENT_TP'])){
        data [row, 'FOLLOW_REC'] = 1
    }
  }

  ### data_out include all columns
  data_out <- data

  ##Output
  # ##Create Output Folder
  outputDir <- file.path(outputBaseFolder, 'output')
  if (!dir.exists(outputDir)){   #Check existence of directory and create it if it doesn't exist
    dir.create(outputDir)
  }
  outputdata_name <- file.path(outputDir, data_name)
  if (!dir.exists(outputdata_name)){ #Check existence of directory and create it if it doesn't exist
    dir.create(outputdata_name)
  }
  newFileName <- paste(c(data_name, '_', method, fold, '.csv', collapse = ""))
  fullPath <- paste(c(outputdata_name,'/',newFileName ), collapse = "")
  write.csv(data_out,fullPath, row.names = FALSE)
  return(fullPath)
}

## Predict the Highest treatment Effect
predict_causal_effect_row <-function(models, recordForEstimate, threshold){
  interRow <- recordForEstimate
  for(i in 1: length(models)){
    treeModel = models[[i]]
    interRow[treeModel$factor] <- predict(treeModel$model, recordForEstimate)
  }
  return (interRow)
}

set.seed(42)  # For reproducibility

projectFolder <- getwd()
outputBaseFolder <- projectFolder


# List of dataset names to iterate over
dataset_names <- c("multi_Trans_ART", "multi_TransNEO", "clin_TransNEO")

# Common parameters
outcome_name <- "resp.pCR"
remove_attributes <- c(
  "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "RCB.score", "RCB.category",
  "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
  "Chemo.second.Taxane", "Chemo.second.Anthracycline",
  "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
)
causal_factors <- c("TP1", "TP2", "TP3", "TP4")

threshold <- 0
method <- "CT"


# --------------------------- Execution Loop ---------------------------

for (data_name in dataset_names) {
  cat("\nProcessing dataset:", data_name, "\n")

  input_data <- read.csv(paste0(projectFolder,'/input/',data_name,'/',data_name, '.csv'))
  input_data <- input_data[complete.cases(input_data[, outcome_name]), ]

  # Create 5-fold cross-validation splits
  folds <- sample(rep(1:5, length.out = nrow(input_data)))

  # Placeholder for all test results
  all_fold_results <- list()


# Main loop over each fold
  for (k in 1:5) {
    cat(sprintf("Processing fold %d...\n", k))

    train_idx <- which(folds != k)
    test_idx <- which(folds == k)

    trainingData <- input_data[train_idx, ]
    testData <- input_data[test_idx, ]

    # Remove attributes
    trainingData <- dplyr::select(trainingData, -all_of(remove_attributes))

    # Train CTR model
    trainedModel <- build_causal_tree_model(data_name, trainingData, causal_factors, outcome_name)

    # Temporarily write test data for consistency
    temp_test_path <- file.path(projectFolder, "input", data_name, paste0(data_name, "_test.csv"))
    dir.create(dirname(temp_test_path), recursive = TRUE, showWarnings = FALSE)
    write.csv(testData, temp_test_path, row.names = FALSE)

    # Generate recommendation
    result_path <- make_binary_recommendation_r(method,
                                                trainedModel,
                                                dirname(temp_test_path),
                                                outputBaseFolder,
                                                data_name,
                                                outcome_name,
                                                remove_attributes,
                                                fold=k)

    # Load fold result and store
    fold_result <- read.csv(result_path)
    fold_result$Fold <- k
    all_fold_results[[k]] <- fold_result
  }

  # Combine all results and write to one CSV
  final_result <- do.call(rbind, all_fold_results)
  final_output_dir <- file.path(outputBaseFolder, "output", data_name)
  dir.create(final_output_dir, showWarnings = FALSE, recursive = TRUE)
  write.csv(final_result, file.path(final_output_dir, paste0(data_name, "_CT_REC.csv")), row.names = FALSE)

  cat("All 5-fold results saved successfully.\n")

}

