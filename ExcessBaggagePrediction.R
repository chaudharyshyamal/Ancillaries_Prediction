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
library(lubridate)
library(caret)
library(usdm)



setwd("C:/Users/mahe/Desktop/Jet_new_data")
data <- as.data.frame(read.table("PredictionDataFinal.csv",header=TRUE,sep=","))

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
data$TripMultiCity<-NULL
data$CardHolderTravelling<-NULL
data$CCPAC<-NULL
data$CCDC<-NULL
data$CCMC<-NULL
data$CCBD<-NULL
data$CCVI<-NULL
data$CCAX<-NULL
data$CCBlank<-NULL



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

data$SeatSelect<-NULL
data$PriorityAdvantage<-NULL
data$JetProtect<-NULL


vifcor(data, th=0.9)

vifstep(data, th=4)


train<-tail(data,75000)
test<-head(data,17743)

#Applying Naive Bayes(25.04)

NaiveBayesModelEB<-naiveBayes(as.factor(ExcessBaggage) ~.,data=train)

NBPredictionEB<-predict(NaiveBayesModelEB,test)

NBEB<-table(NBPredictionEB,test$ExcessBaggage)
confusionMatrix(NBEB)


#Naive Bayes(k fold cross validation)(99.05)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

#x<-head(train,10000)

NaiveBayesCVmodelEB <- train(as.factor(ExcessBaggage) ~.,data=train, 
                           trControl=train_control, method="nb")

NBCVPredictionEB<-predict(NaiveBayesCVmodelEB,test)

NBCVEB<-table(NBCVPredictionEB,test$ExcessBaggage)
confusionMatrix(NBCVEB)


#Using the result above we can ignore columns like SourceWEB, SourceMOBAPP, DeviceDesktop,
#X0to7.days, Day, Direct, TripOW ,SME


#k fold cross validation

#Logistic regression(k fold cross validation)(99.04)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

#x<-head(train,10000)

LogisticRegressionmodelEB <- train(as.factor(ExcessBaggage) ~.,
                                 data=train,trControl=train_control, method="glm")

LRPredictionEB<-predict(LogisticRegressionmodelEB,test)

LREB<-table(LRPredictionEB,test$ExcessBaggage)
confusionMatrix(LREB)


#SVM(k fold cross validation)(99.02)

IsCorporate+TripRT+
  SourceMOBSITE+DeviceAndroid+DeviceApple+
  DeviceOthers+X8to14days+X15to21days+X22to30days+
  X31to60days+X61to90days+Above90days+RevenuePerPax+Night+
  SEM+SEO+Metasearch+Affiliates+SME+Retargetting+
  Newsletter+Social+Male+Child+
  SingleMale+SingleFemale

x<-head(train,5000)

train_control <- trainControl(method="repeatedcv", number=30,repeats = 3)

SVMModelEB<-svm(as.factor(ExcessBaggage) ~.,
              data=x,type="C-classification",scale=TRUE,kernel="polynomial",
              tuneGrid=expand.grid(C=c(.000001, .00001, .0001, .001, .01,
                                       .1,1, 10),sigma=c(.00001, .0001, .001, .01, .1, 1, 10)))


SVMPredictionEB<-predict(SVMModelEB,test)

SVMEB<-table(SVMPredictionEB,test$ExcessBaggage)
confusionMatrix(SVMEB)

#Decision Tree(k fold cross validation)(99.05)

train_control <- trainControl(method="repeatedcv", number=30,repeats=3)

#rpart.grid <- expand.grid(.cp=0.2)

x<-head(data,500)

DecisionTreemodelEB <- train(as.factor(ExcessBaggage) ~IsCorporate+TripRT+
                             SourceMOBSITE+DeviceAndroid+DeviceApple+
                             DeviceOthers+X8to14days+X15to21days+X22to30days+
                             X31to60days+X61to90days+Above90days+RevenuePerPax+Night+
                             SEM+SEO+Metasearch+Affiliates+SME+Retargetting+
                             Newsletter+Social+Male+Child+Couple+CoupleWithChild+
                             SingleMale+SingleFemale+SingleMaleWithChild+
                             SingleFemaleWithChild+MaleWithChild+FemaleWithChild,
                           data=train,method="C5.0Cost",trControl=train_control,
                           tuneGrid = expand.grid(model = "tree", winnow = FALSE,
                                                  trials = 1,cost = 1:10))

#pruned <- prune(DecisionTreemodel, cp = 0.05)

DTPredictionEB<-predict(DecisionTreemodelEB,test,type="raw")

DTEB<-table(DTPredictionEB,test$ExcessBaggage)
confusionMatrix(DTEB)
