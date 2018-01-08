Practical machine learning course assignment
============================================

Synopsis
--------

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement - a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, your goal
will be to use data from accelerometers on the belt, forearm, arm, and
dumbell of 6 participants. They were asked to perform barbell lifts
correctly and incorrectly in 5 different ways. More information is
available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset). Read more:
<http://groupware.les.inf.puc-rio.br/har#ixzz3xsbS5bVX>

This assignment was built up in RStudio, using its knitr functions,
meant to be published in html format. The main goal of the project is to
predict the manner in which 6 participants performed some exercise as
described below. This is the "classe" variable in the training set. The
machine learning algorithm described here is applied to the 20 test
cases available in the test data and the predictions are submitted in
appropriate format to the Course Project Prediction Quiz for automated
grading.

Data processing
---------------

Obtaining data
--------------

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from
<http://groupware.les.inf.puc-rio.br/har>.

### Reading files

    filepath1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    filepath2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

    if(!file.exists("pml-training.csv")) {
      download.file(filepath1, destfile = "pml-training.csv")
    }

    if(!file.exists("pml-testing.csv")) {
      download.file(filepath2, destfile = "pml-testing.csv")
    }
    test <- read.csv("pml-testing.csv", sep = ",", header = TRUE, na.strings=c("NA","#DIV/0!",""))
    training <- read.csv("pml-training.csv", sep = ",", header = TRUE, na.strings=c("NA","#DIV/0!",""))

### Cleaning data

In this section columns that contain NAs or are empty are removed. First
seven features are not numeric (related to the time-series)

    features <- names(test[,colSums(is.na(test)) == 0])[8:46]

    # Since test and training sets are compared, use only features that are present in the testing set.
    training <- training[,c(features,"classe")]
    test <- test[,c(features,"problem_id")]

### Create partitions

For machine training session one must create partitions for it first. In
this case 70% training set and 30% testing set will be created from the
dataset provided.

    library(caret)

    ## Warning: package 'caret' was built under R version 3.4.3

    ## Loading required package: lattice

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 3.4.3

    Partitions  <- createDataPartition(training$classe, p=0.7, list=FALSE)
    Train_set <- training[Partitions, ]
    Test_set  <- training[-Partitions, ]
    dim(Train_set)

    ## [1] 13737    40

    dim(Test_set)

    ## [1] 5885   40

### Correlation analysis

    #Plot
    library(corrgram)

    ## Warning: package 'corrgram' was built under R version 3.4.3

    corrgram(Train_set, order=FALSE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, col.regions=colorRampPalette(c("red","salmon","white","royalblue","navy")), main="Correlation of remaining features")

![](assingment_final_files/figure-markdown_strict/correlations-1.png)
Only few very dark correlation can be found.

Model selection
---------------

Two models will be used in this data analysis: random forests and
decision trees. Confusion matrices are plotted to show their accuracy.

### Random forest analysis on the training dataset

    library(randomForest)

    ## Warning: package 'randomForest' was built under R version 3.4.3

    ## randomForest 4.6-12

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    library(caret)
    set.seed(12345)
    random_forest_model <- randomForest(classe ~ ., data = Train_set, ntree = 1000, na.action =na.exclude)

    #Use prediction on testing set of training set result
    prediction <- predict(random_forest_model, Test_set, type = "class")
    confusionMatrix(prediction, Test_set$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1670   14    0    0    0
    ##          B    3 1122   12    0    1
    ##          C    0    3 1012   20    3
    ##          D    1    0    2  944    4
    ##          E    0    0    0    0 1074
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9893          
    ##                  95% CI : (0.9863, 0.9918)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9865          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9976   0.9851   0.9864   0.9793   0.9926
    ## Specificity            0.9967   0.9966   0.9946   0.9986   1.0000
    ## Pos Pred Value         0.9917   0.9859   0.9750   0.9926   1.0000
    ## Neg Pred Value         0.9990   0.9964   0.9971   0.9959   0.9983
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2838   0.1907   0.1720   0.1604   0.1825
    ## Detection Prevalence   0.2862   0.1934   0.1764   0.1616   0.1825
    ## Balanced Accuracy      0.9971   0.9909   0.9905   0.9889   0.9963

### Out-of-sample error for random forest

    sum(prediction != Test_set$classe) / length(Test_set$classe)

    ## [1] 0.01070518

It seems that the out-of-sample error is **0.22 % ** for the training
subset "Test\_set"

### Random forest analysis result on the testing dataset

    predict(random_forest_model, test, type = "class")

    ##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
    ##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
    ## Levels: A B C D E

### Train a decision tree model

    library(rattle)

    ## Warning: package 'rattle' was built under R version 3.4.3

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.1.0 Copyright (c) 2006-2017 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    ## 
    ## Attaching package: 'rattle'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

    library(rpart)

    ## Warning: package 'rpart' was built under R version 3.4.3

    #Create random sample without NAs
    set.seed(55555)
    #training <- Train_set[ , colSums(is.na(Train_set)) == 0]
    rand_sample <- Train_set[sample(1:nrow(Train_set), 100, replace = FALSE), ]

    #Train random sample using decision tree with principal componend analysis
    dt_mod <- rpart(classe~., data=rand_sample, method="class")
    fancyRpartPlot(dt_mod)

![](assingment_final_files/figure-markdown_strict/dt-1.png)

Decision tree model shows somewhat equal prediction amongst classes.

### Use decision tree on test-set and plot a confusion matrix

    dt_model <- rpart(classe~., data=Test_set, method="class")
    prediction_dt <- predict(dt_model, Test_set, type = "class")
    confusionMatrix(prediction_dt, Test_set$classe)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1505  247  243  109   94
    ##          B   35  692   57   30  110
    ##          C   14   51  552  149   61
    ##          D   87   96  138  660   95
    ##          E   33   53   36   16  722
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.702           
    ##                  95% CI : (0.6901, 0.7136)
    ##     No Information Rate : 0.2845          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.6189          
    ##  Mcnemar's Test P-Value : < 2.2e-16       
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.8990   0.6076   0.5380   0.6846   0.6673
    ## Specificity            0.8354   0.9511   0.9434   0.9155   0.9713
    ## Pos Pred Value         0.6847   0.7489   0.6675   0.6134   0.8395
    ## Neg Pred Value         0.9542   0.9099   0.9063   0.9368   0.9284
    ## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
    ## Detection Rate         0.2557   0.1176   0.0938   0.1121   0.1227
    ## Detection Prevalence   0.3735   0.1570   0.1405   0.1828   0.1461
    ## Balanced Accuracy      0.8672   0.7793   0.7407   0.8001   0.8193

### Out-of-sample error for decision tree

    sum(prediction_dt != Test_set$classe) / length(Test_set$classe)

    ## [1] 0.2980459

20 different test case prediction
---------------------------------

    predict_test <- predict(random_forest_model, newdata=test)
    test$problem_id <- predict_test
    result <- data.frame(problem_id = test$problem_id, classe = predict_test)
    result[1:20,]

    ##    problem_id classe
    ## 1           B      B
    ## 2           A      A
    ## 3           B      B
    ## 4           A      A
    ## 5           A      A
    ## 6           E      E
    ## 7           D      D
    ## 8           B      B
    ## 9           A      A
    ## 10          A      A
    ## 11          B      B
    ## 12          C      C
    ## 13          B      B
    ## 14          A      A
    ## 15          E      E
    ## 16          E      E
    ## 17          A      A
    ## 18          B      B
    ## 19          B      B
    ## 20          B      B

Results
-------

As you can see from the confusion matrix of the rando forest, the
accuracy is very high (99.78%) with very high confidence intervals and
low P-values. The resulting low out-of-sample error (0.22%) seems too
good to be true. However decision tree results show 98.11 % accuracy
with equally low P-values while out-of-sample error is higher (1.89%).
This shows that random forest is a better option in this case for model
selection.
