#set working directory and read data 
library(foreach)
library(lubridate)
library(chron)
library(timeDate)
library(datetime)
library(date)
library(tidyverse)
library(caret)
library(rpart)
library(dplyr)
library(lme4)
library(usdm)
library(car)
library(SparseM)
library(xgboost)
library(readr)
library(stringr)
library(rpart)
library(C50)
library(mlr)
library(e1071)

library(e1071)
library(lubridate)
library(caret)
library(usdm)
library(ROSE)



setwd("C:/Users/mahe/Desktop/Jet_new_data/PredictionFinal")
data <- as.data.frame(read.table("Sep17.csv",header=TRUE,sep=","))

revenueperpax<-0
#Creating a new column RevenuePerPax
for (i in 1:nrow(data))
{
  revenueperpax[i]<-data$Revenue[i]/data$Pax[i]  
}


data$RevenuePerPax <-revenueperpax

#As d has both domestic and international data,here data has only domestic data
data <- subset(data, IsInternational==0)

#Convert timestamp(date and time) to a form which R can understand
mydate <- mdy_hms(data$FirstDepartureDate)

#Here we can get any(hour,minute,second,date,month,year) by using function unclass
x<-unclass(as.POSIXlt(mydate))$hour

#Here putting condition to separate time into morning,afternoon,evening,night
data$Day <- ifelse(x>=6 & x<=18 ,1,0)

data$Night<- ifelse((x>=0 & x<6) | (x>18),1,0)



#delete/remove column(removed TripMulticity,CardHolderTravelling and
#all those card types)



#Deleted IsInternational column
data$IsInternational<-NULL


#Making and adding different columns for Referrer types

data$Direct <- ifelse(data$ReferrerCategory=="Direct", 1,0)
data$SEM <- ifelse(data$ReferrerCategory=="SEM", 1,0)
data$SEO <- ifelse(data$ReferrerCategory=="SEO", 1,0)
data$Metasearch <- ifelse(data$ReferrerCategory=="Metasearch", 1,0)
data$Affiliates <- ifelse(data$ReferrerCategory=="Affiliates", 1,0)
data$SME <- ifelse(data$ReferrerCategory=="SME", 1,0)
data$Retargetting <- ifelse(data$ReferrerCategory=="Retargetting", 1,0)
data$Newsletter <- ifelse(data$ReferrerCategory=="Newsletter", 1,0)
data$Social <- ifelse(data$ReferrerCategory=="Social", 1,0)

data$Male <- ifelse((data$Mr>0 | data$Mrs>0), 1,0)
data$Female <- ifelse((data$Miss>0 | data$Ms>0), 1,0)
data$Child <- ifelse(data$Infant>0, 1,0)
data$Couple <- ifelse(((data$Mr>0 | data$Mrs>0) && (data$Miss>0 | data$Ms>0)), 1,0)
data$CoupleWithChild <- ifelse(((data$Mr>0 | data$Mrs>0) &&
                                  (data$Miss>0 | data$Ms>0) && data$Infant>0), 1,0)
data$SingleMale <- ifelse((data$Mr==1 | data$Mrs==1), 1,0)
data$SingleFemale <- ifelse((data$Miss==1 | data$Ms==1), 1,0)
data$SingleMaleWithChild <- ifelse(((data$Mr==1 | data$Mrs==1) && data$Infant>0), 1,0)
data$SingleFemaleWithChild <- ifelse(((data$Miss==1 | data$Ms==1) && data$Infant>0), 1,0)
data$MaleWithChild <- ifelse(((data$Mr>0 | data$Mrs>0) && data$Infant>0), 1,0)
data$FemaleWithChild <- ifelse(((data$Miss>0 | data$Ms>0) && data$Infant>0), 1,0)


data$Mr<-NULL
data$Mrs<-NULL
data$Ms<-NULL
data$Miss<-NULL
data$Dr<-NULL
data$MSTR<-NULL
data$Prof<-NULL
data$Capt<-NULL
data$Infant<-NULL


data$BookingTimeStamp<-NULL
data$BookingReloc<-NULL
data$Revenue<-NULL
data$Pax<-NULL
data$JPPax<-NULL
data$ReferrerCategory<-NULL
data$FirstDepartureDate<-NULL


train<-tail(data,80000)
test<-head(data,27361)

#Oversampling

data_balanced_over <- ovun.sample(SeatSelect ~ ., data = train,
                                  method = "over",N = 100000)$data

#Undersampling

data_balanced_under <- ovun.sample(SeatSelect ~ ., data = train,
                                  method = "under",N = 30000)$data

sample <- sample.int(n = nrow(data), size = floor(.8*nrow(data)), replace = F)
train <- data[sample, ]
test  <- data[-sample, ]

#Applying Naive Bayes(74.89)(74.12)
  
NaiveBayesModel<-naiveBayes(as.factor(SeatSelect) ~.,
                              data=train)

NBPrediction<-predict(NaiveBayesModel,test)

NB<-table(NBPrediction,test$SeatSelect)
confusionMatrix(NB)

saveRDS(NaiveBayesModel, "./NaiveBayesModel.rds")


#Checking collinearity(if the result comes out to be greater than 4 as threshold is set as 4
#then we can exclude those columns) (>4 then those columns are collinear)

dummy_df<-data[,c(1:17,22:44)]

vifcor(dummy_df, th=0.9)

vifstep(dummy_df, th=4)


#Using the result above we can ignore columns like SourceWEB, SourceMOBAPP, DeviceDesktop,
#X0to7.days, Day, Direct, TripOW ,SME
 

#k fold cross validation

#Logistic regression(k fold cross validation)(86.73)(84.96)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

x<-head(train,10000)

LogisticRegressionmodel <- train(as.factor(SeatSelect) ~.,
                                 data=train,trControl=train_control, method="glm")

LRPrediction<-predict(LogisticRegressionmodel,test)

LR<-table(LRPrediction,test$SeatSelect)
confusionMatrix(LR)



#Naive Bayes(k fold cross validation)(86.78)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

x<-head(train,10000)

NaiveBayesmodel <- train(as.factor(SeatSelect) ~.,
                           data=train, 
                           trControl=train_control, method="nb")

NBPrediction<-predict(NaiveBayesmodel,test)

NB<-table(NBPrediction,test$SeatSelect)
confusionMatrix(NB)


#SVM(k fold cross validation)(86.78 will try)(84.28 will try)

x<-head(train,100)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

SVMModel<-svm(as.factor(SeatSelect) ~.,data=train,
                type="C-classification",scale=TRUE,kernel="polynomial",
                trControl=train_control,
                tuneGrid=expand.grid(C=c(.000001, .00001, .0001, .001, .01,
                .1,1, 10),sigma=c(.00001, .0001, .001, .01, .1, 1, 10)))
                    

SVMPrediction<-predict(SVMModel,test)

SVM<-table(SVMPrediction,test$SeatSelect)
confusionMatrix(SVM)

#Decision Tree(k fold cross validation)(85.08)

train_control <- trainControl(method="repeatedcv", number=30,repeats=3)

#rpart.grid <- expand.grid(.cp=0.2)

DecisionTreemodel <- train(as.factor(SeatSelect) ~.,
                             data=train,method="C5.0Cost",trControl=train_control,
                             tuneGrid = expand.grid(model = "tree", winnow = FALSE,
                                                    trials = 1,cost = 1:10))
                           
#pruned <- prune(DecisionTreemodel, cp = 0.05)

DTPrediction<-predict(DecisionTreemodel,test,type="raw")

DT<-table(DTPrediction,test$SeatSelect)
confusionMatrix(DT)











