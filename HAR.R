library(caret)
library(neuralnet)
library(plyr)

load('activity_data.RData')
set.seed(100)

# Extract necessary features
#label_index <- c(1:8, 43:48, 83:85, 123:128, 163:165)
label_index <- c(1:563)
data <- activity_data[, label_index]
#data$label <- as.factor(data$label)
data$subject <- NULL

# Load .RData file into environment.
loadRData <- function(fileName){
  load(fileName)
  get(ls()[ls() != "fileName"])
}

# Save results into local files.
save_all <- function() {
  save(nn_ret,    file="./CleanData/ret/nn_ret.RData")
}

train <- NULL
test <- NULL
inTrain <- createDataPartition(y=data$label, p=0.7, list=FALSE)
train <- data[inTrain, ]
test <- data[-inTrain, ]
# The lists to store backprop performance
nn_bkp <- NULL    #neural network backkpropagation
nn_bkp_pred <- NULL #backpropograted model
#feature names assigned to be used in formula in later step
colnames(train) <- paste("Feature", colnames(train), sep = "_")
colnames(train)[1] <- "label"
n <- names(train)
for_mula <- as.formula(paste("label ~", paste(n[!n %in% "label"], collapse = " + ")))
for_mula <- as.formula(for_mula)
nn <- neuralnet(for_mula,data=train,hidden=c(10,6), act.fct = "tanh", err.fct="sse", algorithm = "backprop", learningrate = 0.01, linear.output=FALSE,lifesign = "none")
nn_bkp <- nn
nn_bkp_pred<-prediction(nn, list.glm = NULL)
save(nn_bkp_pred,    file="nn_bkp_pred.RData")
save(nn_bkp,    file="nn_bkp.RData")