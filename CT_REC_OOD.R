###07/05/2025 Make Recommendation from trained models

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
  # susanathey/causalTree, quiet = TRUE)

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



# source("binary_models.R")
# projectFolder = getwd()
# print(projectFolder)
# outputBaseFolder = projectFolder
# output_folder=outputBaseFolder
#
# data_name = 'OOD_multi_Trans_ART'
# causal_factors = c('TP1','TP2','TP3','TP4')
#
# # data_name = 'OOD_multi_3TS_Trans_ART'
# # causal_factors = c('TP1','TP2','TP3')
#
# outcome_name = 'resp.pCR'
# remove_attributes = c('Trial.ID','resp.Chemosensitive','resp.Chemoresistant','RCB.score','RCB.category','Chemo.NumCycles','Chemo.first.Taxane','Chemo.first.Anthracycline', 'Chemo.second.Taxane', 'Chemo.second.Anthracycline', 'Chemo.any.Anthracycline','Chemo.any.antiHER2')
#
# black_list = NULL
# threshold <- -201
#
# input_data_folder = paste(projectFolder,'/input/',data_name,sep='')
# method = "CT"
# train_data = paste (data_name, '_train', sep='')        ##'OOD_multi_Trans_ART_train'
# test_data = paste (data_name, '_test', sep='')      ##'OOD_multi_Trans_ART_test'
#
#
#
# trainingFile<- paste (input_data_folder, '/', train_data, '.csv',sep='')
# trainingData <-read.csv(file = trainingFile)
#
#
# trainingData <- dplyr::select (trainingData, -remove_attributes)

# # Define project folder
# projectFolder <- getwd()
#
# # Configuration for multiple datasets
# datasets <- list(
#   list(
#     data_name = "OOD_multi_Trans_ART",
#     causal_factors = c("TP1", "TP2", "TP3", "TP4")
#   ),
#   list(
#     data_name = "OOD_multi_3TS_Trans_ART",
#     causal_factors = c("TP1", "TP2", "TP3")
#   )
# )
#
# # Common parameters
# outcome_name <- 'resp.pCR'
# remove_attributes <- c(
#   'Trial.ID', 'resp.Chemosensitive', 'resp.Chemoresistant',
#   'RCB.score', 'RCB.category', 'Chemo.NumCycles',
#   'Chemo.first.Taxane', 'Chemo.first.Anthracycline',
#   'Chemo.second.Taxane', 'Chemo.second.Anthracycline',
#   'Chemo.any.Anthracycline', 'Chemo.any.antiHER2'
# )
# black_list <- NULL
# threshold <- -201
# method <- "CT"
#
# # Loop through datasets
# for (cfg in datasets) {
#   data_name <- cfg$data_name
#   causal_factors <- cfg$causal_factors
#
#   input_data_folder <- paste0(projectFolder, '/input/', data_name)
#   # train_data <- paste0(data_name, '_train')
#   # test_data <- paste0(data_name, '_test')
#
#   trainingFile <- file.path(input_data_folder, paste0(data_name, '_train.csv'))
#   trainingData <- read.csv(file = trainingFile)
#
#   # Remove specified attributes
#   trainingData <- dplyr::select(trainingData, -one_of(remove_attributes))
#
#   cat("\nProcessed training data for:", data_name, "\n")
#   cat("Included causal factors:", paste(causal_factors, collapse = ", "), "\n")
#   # You can now proceed with further analysis or model training
#
#   trainedModel <- build_causal_tree_model(data_name, trainingData, causal_factors, outcome_name)
#
#
#
#   make_binary_recommendation_r(method,
#                              trainedModel,
#                              input_data_folder,
#                              outputBaseFolder,
#                              data_name,
#                              outcome_name,
#                              remove_attributes)
#
#
# }
# Define project folder



run_multiple_ct_models <- function(projectFolder, datasets, outcome_name, remove_attributes,
                                   method = "CT", threshold = 0, outputBaseFolder = NULL) {
  if (is.null(outputBaseFolder)) {
    outputBaseFolder <- projectFolder
  }

  for (cfg in datasets) {
    data_name <- cfg$data_name
    causal_factors <- cfg$causal_factors

    cat("\n--- Running Causal Tree for:", data_name, "---\n")

    input_data_folder <- file.path(projectFolder, 'input', data_name)
    training_file <- file.path(input_data_folder, paste0(data_name, '_train.csv'))

    # Load and preprocess training data
    training_data <- read.csv(training_file)
    training_data <- dplyr::select(training_data, -one_of(remove_attributes))

    cat("Included causal factors:", paste(causal_factors, collapse = ", "), "\n")

    # Train model
    trainedModel <- build_causal_tree_model(data_name, training_data, causal_factors, outcome_name)

    # # Generate recommendations
    make_binary_recommendation_r(method,
                             trainedModel,
                             input_data_folder,
                             outputBaseFolder,
                             data_name,
                             outcome_name,
                             causal_factors,
                             threshold,
                             remove_attributes)
  }
}



## # build Causal Tree model
build_causal_tree_model <- function(data_name, training_data, causal_factors, outcome_name, splitRule = 'CT', cvRule = 'CT') {
  results <- list()
  output_folder = getwd()
  out_put <- file.path(output_folder, 'output')

  if (!dir.exists(out_put)) {
    dir.create(out_put)
  }
  outputDir <- file.path(out_put, 'Trained-CTRModel')

  if (!dir.exists(outputDir)) {
    dir.create(outputDir)
  }

  for (i in 1:length(causal_factors)) {
    reg <- glm(as.formula(paste(causal_factors[[i]], ' ~ . -', outcome_name, sep = "")),
               family = binomial,
               data = training_data)

    propensity_scores <- reg$fitted
    tree <- causalTree(as.formula(paste(outcome_name, ' ~ . ', sep = "")),
                       data = training_data,
                       treatment = training_data[[causal_factors[[i]]]],
                       split.Rule = splitRule,
                       cv.option = cvRule,
                       split.Honest = TRUE,
                       cv.Honest = TRUE,
                       split.Bucket = FALSE,
                       xval = 5,
                       cp = 0,
                       minsize = 3L,
                       propensity = propensity_scores)


    # tree1 <- causalTree(as.formula(paste(outcomeName, ' ~. ', sep= ""))
    #                     , data = trainingData, treatment = trainingData[[causalFactor]],
    #                     split.Rule = splitRule, cv.option = cvRule,
    #                     split.Honest = T, cv.Honest = T, split.Bucket = F,
    #                     xval = 5, cp = 0, propensity = propensity_scores)


    opcp <- tree$cptable[, 2][which.min(tree$cptable[, 4])]
    opfit <- prune(tree, opcp)


    treeFileName <- paste(data_name, causal_factors[[i]], '_tree.png', sep = '')
    treeFile <- file.path(outputDir, treeFileName)
    png(file = treeFile, width = 1200, height = 900)

    rpart.plot(opfit)
    dev.off()

    treeModel <- list()
    treeModel$model <- opfit
    treeModel$factor <- causal_factors[[i]]
    results <- append(results, list(treeModel))

    trainingModel <- opfit
    trainedModelFileName <- paste(data_name, causal_factors[[i]], '_trainedModel.RDS', sep = '')

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
                                         causal_factors,
                                         threshold,
                                         remove_attributes = NULL) {
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
  # data_out['RealOUTCOME'] <- data[outcome_name]


  ##Output
  # ##Create Output Folder
  outputDir <- file.path(outputBaseFolder, 'output')
  if (!dir.exists(outputDir)){   #Check existence of directory and create it if it doesn't exist
    dir.create(outputDir)
  }
  # methodFolder <- file.path(outputDir, method)
  # if (!dir.exists(methodFolder)){   #Check existence of directory and create it if it doesn't exist
  #   dir.create(methodFolder)
  # }
  outputdata_name <- file.path(outputDir, data_name)
  if (!dir.exists(outputdata_name)){ #Check existence of directory and create it if it doesn't exist
    dir.create(outputdata_name)
  }

  newFileName <- paste(c(data_name, '_', method, '_REC.csv', collapse = ""))
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



projectFolder <- getwd()

# Dataset configurations
datasets <- list(
  list(data_name = "OOD_multi_Trans_ART", causal_factors = c("TP1", "TP2", "TP3", "TP4")),
  list(data_name = "OOD_multi_3TS_Trans_ART", causal_factors = c("TP1", "TP2", "TP3"))
)

# Shared parameters
outcome_name <- "resp.pCR"
remove_attributes <- c(
  "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant",
  "RCB.score", "RCB.category", "Chemo.NumCycles",
  "Chemo.first.Taxane", "Chemo.first.Anthracycline",
  "Chemo.second.Taxane", "Chemo.second.Anthracycline",
  "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
)

# Run the function
run_multiple_ct_models(
  projectFolder = projectFolder,
  datasets = datasets,
  outcome_name = outcome_name,
  remove_attributes = remove_attributes
)