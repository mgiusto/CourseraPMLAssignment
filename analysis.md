Coursera Practical Machine Learning Assignment
========================================================
# Introduction
Aim of the project is to build a classifier to understand if a human is performing correctly barbell lifts, a typical gym exercise.
Data were collected by http://groupware.les.inf.puc-rio.br/har .

In the following sections it is reported the code used for loading the data, selecting variables and building the classifier.


``` 
## Warning: package 'caret' was built under R version 3.1.2
```


# Downloading data
Data can be downloaded from with the following code:

```r
if (!file.exists("training.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
        destfile = "training.csv")
}
if (!file.exists("testing.csv")) {
    download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
        destfile = "testing.csv")
}
```


# Loading data, exploratory analysis and variable selection
It is necessary to select the variables to use for the predictive model. Variables not related to belt, dumbbell, arm and forearm can be discarded. On the remainings, it is possible to mantain only the ones that have a good variance and represent most of the variaty of the entire dataset.

```r
training = read.csv("training.csv")
output = training$classe
training2 = training[, grepl("belt|dumbbell|arm|forearm", colnames(training))]
nzv = nearZeroVar(training2, saveMetrics = TRUE)
training3 = training2[, nzv$nzv == FALSE]
countNA = rapply(training3, function(x) sum(is.na(x)))
training4 = training3[, countNA < dim(training3)[1] * 0.8]
preProcPCA = preProcess(training4, method = "pca", thresh = 0.9)
training5 = predict(preProcPCA, training4)
```


# Building and evaluating model
To build a classifier it is possible to use different algorithms. In the following, it has been chosen to build a random forest with cross validation on 10 folds.


```r
train = training5
train$output = output
fitControl <- trainControl(method = "cv", number = 10)
model = train(output ~ ., method = "rf", data = train, trControl = fitControl)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.2
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
model
```

```
## Random Forest 
## 
## 19622 samples
##    19 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## 
## Summary of sample sizes: 17659, 17660, 17660, 17660, 17659, 17659, ... 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy  Kappa  Accuracy SD  Kappa SD
##    2    1         1      0.003        0.004   
##   10    1         1      0.003        0.004   
##   19    1         1      0.003        0.004   
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 1.72%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 5547   11   15    3    4    0.005914
## B   44 3719   32    0    2    0.020543
## C    6   37 3351   24    4    0.020748
## D    4    3  102 3103    4    0.035137
## E    0    9   20   13 3565    0.011644
```


Due to cross-validation, more random forest classifiers are generated. The final model, that is the best one among the generated, is composed of 500 trees, the expected error rate is 1.72%. From the confusion matrix it can be seen that classe A is very well predicted, on the others there is a greater error, but always acceptably low.

# Applying model to new data
To apply the model to new data, once loaded it is necessary to select only the variable used to build the model, then it is possible to generate prediction.

```r
testing = read.csv("testing.csv")
testing = testing[, colnames(training4)]
test = predict(preProcPCA, testing)
predict(model, test)
```

```
##  [1] B A A A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


