library(dplyr)
library(ggplot2)
library(tidyr)
library(SDMTools)
library(readr)
library(digest)
library(ISLR)
library(car) 
library(leaps)
library( Matrix)
library(foreach)
library(glmnet)
library(gridExtra)
library(lsmeans)
library(limma)
library(Sleuth3)
library(tseries)
library(forecast)
library(ggplot2)
library(MASS)
library(mvtnorm)
library(epitools)
library(samplesizeCMH)
library(caret)
library(GGally)
library(glmnet)
library(bestglm)
library(data.table)
library(broom)
library(plyr)
library(repr)
library(ResourceSelection)
library(ROCR)
library(pROC)
library(FNN)

#install.packages('limma')

set.seed(1234)
bank_20 = read.csv("data/bank-additional/bank-additional-full.csv", sep=";")
head(bank_20)
hist(bank_20$pdays)
temp = bank_20 %>% filter(pdays != 999)
dim(temp)
hist(temp$pdays)
summary(bank_20)

#b <- bank_20 %>% filter(default == 'yes')
#bank_20 <- bank_20 %>% filter(default != 'yes')

clean_bank_20<-bank_20
yes_indices = which(clean_bank_20$y == "yes")
yes_train_indices = sample(yes_indices, length(yes_indices) * .9)
no_indices = which(clean_bank_20$y == "no")
no_train_indices = sample(no_indices, length(yes_indices))
train_indices = c(no_train_indices,yes_train_indices)

balanced_train_bank_20 = clean_bank_20[train_indices,]
balanced_train_bank_20<-balanced_train_bank_20
summary(balanced_train_bank_20)
test_bank_20 = clean_bank_20[-train_indices,]
test_bank_20<-test_bank_20
summary(test_bank_20)
#test_bank_20$default <- as.factor(test_bank_20$default)

#balanced_train_bank_20 <- rbind(balanced_train_bank_20,b[3,])
#test_bank_20 <- rbind(test_bank_20,b[1:2,])


dat.test.y<- test_bank_20$y
dat.test.y<-ifelse(dat.test.y == "yes", 1,0)
dat.test.x<- model.matrix(y~ .,test_bank_20)[,-1]
##############################################################################################################
# Stepwise selection 
full.model <- glm(y ~., data = balanced_train_bank_20, family = binomial)


# Stepwise regression model for AIC  ##########
step.model <- stepAIC(full.model, direction = "both", 
                      trace = FALSE)
summary(step.model)    # The stars in the parenthesis means that it is involved in the model.
vif(step.model) # vif for stepwise

##### Predict Stepwise ##################################
step.predict <-predict(step.model, newdata=test_bank_20)


perfStep <- performance(prediction(step.predict, dat.test.y), 'tpr', 'fpr')            ######## start here ######### 
plot(perf,main="Stepwise")

step.predict <- ifelse(step.predict > 0.5, 'yes','no')
step.predict <- as.factor(step.predict)

table(test_bank_20$y,step.predict)
cfm = confusionMatrix(step.predict,test_bank_20$y)
cfm  # for Stepwise


############ forward regression model for AIC  #############
step.model <- stepAIC(full.model, direction = "forward", 
                      trace = FALSE)

model.main<- glm(y~ ., balanced_train_bank_20,family = binomial(link="logit"))
model.null<-glm(y ~ 1, data=balanced_train_bank_20,family = binomial(link="logit"))
step.model <- step(model.null,
     scope = list(upper=model.main),
     direction="forward",
     test="Chisq",
     data=balanced_train)
summary(step.model) #Forward selection
vif(step.model) # vif for forward
alias(step.model)

##### Predict Forward ##################################
step.predict <-predict(step.model, newdata=test_bank_20)


perfForward <- performance(prediction(step.predict, dat.test.y), 'tpr', 'fpr')            ######## start here ######### 
plot(perf,main="Forward")

step.predict <- ifelse(step.predict > 0.5, 'yes','no')
step.predict <- as.factor(step.predict)

table(test_bank_20$y,step.predict)
cfm = confusionMatrix(step.predict,test_bank_20$y)
cfm   # for forward


########## Backward regression model For AIC ######################
step.model <- stepAIC(full.model, direction = "backward", 
                          trace = FALSE)
summary(step.model) #backward selection
vif(step.model) # vif for backward
alias(step.model)

##### Predict Backward ##################################
step.predict <-predict(step.model, newdata=test_bank_20)

perfback <- performance(prediction(step.predict, dat.test.y), 'tpr', 'fpr')            ######## start here ######### 
plot(perf,main="Backward")

step.predict <- ifelse(step.predict > 0.5, 'yes','no')
step.predict <- as.factor(step.predict)

table(test_bank_20$y,step.predict)
cfm = confusionMatrix(step.predict,test_bank_20$y)
cfm   # for Backward


# #################################   Lasso    ##########################################
x <- model.matrix(y~ .,balanced_train_bank_20)[,-1]
#dat.train.x <- as.matrix(balanced_train_bank_20)
y <- ifelse(balanced_train_bank_20$y == "yes", 1, 0)
#dat.train.y <- balanced_train_bank_20$y
cvfit <- cv.glmnet(x, y, family = "binomial", type.measure = "class", nlambda = 1000)
summary(cvfit)

plot(cvfit)
coef(cvfit, s = "lambda.min")
#CV misclassification error rate 
cvfit$cvm[which(cvfit$lambda==cvfit$lambda.min)]

#Optimal penalty
cvfit$lambda.min


#########
finalmodel<-glmnet(x, y, family = "binomial",lambda=cvfit$lambda.min)
vif(finalmodel)
########### Predict Lasso ##############
lasso.predict  <- predict(finalmodel, s = cvfit$lambda.min, newx = dat.test.x)

perflasso <- performance(prediction(lasso.predict, dat.test.y), 'tpr', 'fpr')            ######## start here ######### 
plot(perf,main="LASSO")

lasso.predict <- ifelse(lasso.predict > 0.5, 'yes','no')
lasso.predict <- as.factor(lasso.predict)

table(test_bank_20$y,lasso.predict)
cfm = confusionMatrix(lasso.predict,test_bank_20$y)
cfm   # for Lasso


#### ROC Plot ##########
plot(perfForward)
plot(perfStep,col="orange", add = TRUE)
plot(perfback,col="blue", add = TRUE)
plot(perflasso,col="yellow", add = TRUE)
legend("bottomright",legend=c("Forward","Stepwise","Backward", "Lasso"),col=c("black","orange","blue","yellow"),lty=1,lwd=1)
abline(a=0, b= 1)



############# Logistic regression Fabio's final model #############################
model<-glm(y ~ job + education + default + contact +duration + poutcome + pdays + campaign, family = binomial(link = "logit"),data = balanced_train_bank_20)
logr.probs<-predict(model, newdata=test_bank_20)

logr.probs<- ifelse(logr.probs > 0.5, 'yes','no')
logr.probs <- as.factor(logr.probs)

table(test_bank_20$y,logr.probs)
cfm = confusionMatrix(logr.probs,test_bank_20$y)
cfm

################ logistic regression final ############
model<-glm(y ~ job + default + contact + duration + month + poutcome + campaign, family = binomial(link = "logit"),data = balanced_train_bank_20)
logr.probs<-predict(model, newdata=test_bank_20)

logr.probs<- ifelse(logr.probs > 0.5, 'yes','no')
logr.probs <- as.factor(logr.probs)

table(test_bank_20$y,logr.probs)
cfm = confusionMatrix(logr.probs,test_bank_20$y)
cfm

plot(predict(model),residuals(model),col=c("blue","red")[1+y])
abline(h=0,lty=2,col="grey")
plot(cooks.distance(model))
#plot(cooks.distance(step.model, infl = lm.influence(step.model, do.coef = FALSE), res = weighted.residuals(step.model), sd = sqrt(deviance(step.model)/df.residual(step.model))))


########################New Fabio Model##############################################################
model <- glm(y ~ job + education + marital + housing + contact + default + month + day_of_week + duration + campaign + pdays + poutcome + emp.var.rate + cons.conf.idx + cons.price.idx + nr.employed, family = binomial(link = "logit"),data = balanced_train_bank_20)
logr.probs<-predict(model, newdata=test_bank_20)

logr.probs<- ifelse(logr.probs > 0.5, 'yes','no')
logr.probs <- as.factor(logr.probs)

table(test_bank_20$y,logr.probs)
cfm = confusionMatrix(logr.probs,test_bank_20$y)
cfm
###############################Complex Logistic ###################################
alt_pdays_cat_train = ifelse(balanced_train_bank_20$pdays == 999, 0, 1)
alt_clean_bank_20<- balanced_train_bank_20
alt_clean_bank_20$pdays_cat = alt_pdays_cat_train

alt_pdays_cat_test = ifelse(test_bank_20$pdays == 999, 0, 1) 
alt_clean_bank_20_test<- test_bank_20
alt_clean_bank_20_test$pdays_cat = alt_pdays_cat_test

model <- glm(y ~ job + default + contact + month + duration + campaign + pdays_cat +poutcome +  job*default + contact*duration + pdays_cat*contact + pdays*duration, family = binomial(link = "logit"),data = alt_clean_bank_20)

logr.probs<-predict(model, newdata=alt_clean_bank_20_test)

logr.probs<- ifelse(logr.probs > 0.5, 'yes','no')
logr.probs <- as.factor(logr.probs)

table(alt_clean_bank_20_test$y,logr.probs)
cfm = confusionMatrix(logr.probs,test_bank_20$y)
cfm$byClass[1]

############# Compare with KNN ########################
#library(class)

splitPerc = .8
iterations = 50
numks = 50
masterAcc = matrix(nrow = iterations, ncol = numks)
masterSens = matrix(nrow = iterations, ncol = numks)

clean_bank_20$contactE = ifelse(clean_bank_20$contact == 'cellular' , 1, 0)
clean_bank_20$pdaysE = ifelse(clean_bank_20$pdays == 999, 0, 1)


for(j in 1:iterations) {
  set.seed(Sys.time())
  #accs = data.frame(accuracy = numeric(numks), k = numeric(numks))
  #sens = data.frame(sensitivity = numeric(numks), k = numeric(numks))
  trainIndices = sample(1:dim(clean_bank_20)[1],round(splitPerc * dim(clean_bank_20)[1]))
  train = clean_bank_20[trainIndices,]
  test = clean_bank_20[-trainIndices,]
  for(i in 1:numks) {
    classifications = knn(train[,c('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE')],test[,c('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE')],as.factor(train$y), prob = TRUE, k = i)
    #table(as.factor(test$Attrition),classifications)
    CM = confusionMatrix(classifications,test$y)
    masterAcc[j,i] = CM$overall[1]
    masterSens[j,i] = CM$byClass[1]
  }
}

system(Sys.time(), intern = TRUE)

MeanAcc = colMeans(masterAcc)
MeanSens = colMeans(masterSens)

#plot(seq(1,numks,1),MeanAcc, type = "l")
#plot(seq(1,numks,1),MeanSens, type = "l")

kACC = which.max(MeanAcc)
kSens = which.max(MeanSens)

set.seed(1234)
yes_indices = which(clean_bank_20$y == "yes")
yes_train_indices = sample(yes_indices, length(yes_indices) * .9)
no_indices = which(clean_bank_20$y == "no")
no_train_indices = sample(no_indices, length(yes_indices))
train_indices = c(no_train_indices,yes_train_indices)

balanced_train_bank_20 = clean_bank_20[train_indices,]
balanced_train_bank_20<-balanced_train_bank_20
summary(balanced_train_bank_20)
test_bank_20 = clean_bank_20[-train_indices,]
test_bank_20<-test_bank_20

balancetrainNew <- balanced_train_bank_20 %>% dplyr::select('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE','y')
testNew <- test_bank_20 %>% dplyr::select('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE','y')
classifications = knn(balanced_train_bank_20[,c('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE')],test_bank_20[,c('age', 'duration', 'emp.var.rate', 'campaign','cons.price.idx','cons.conf.idx','euribor3m','nr.employed','contactE','pdaysE')],as.factor(balanced_train_bank_20$y), prob = TRUE, k = kSens)
table(test_bank_20$y,classifications)
cfm = confusionMatrix(classifications,test_bank_20$y)
cfm

################ Compare with SVM with KNN predictors###############
library(e1071)
model <- svm(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm

################### Compare with Naive Bayes with KNN predictors############
model <- naiveBayes(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm


################ Compare with SVM with logistic complex predictors###############
balancetrainNew <- balanced_train_bank_20 %>% dplyr::select('job', 'education' , 'marital' , 'housing' , 'contact' , 'default' ,'month' , 'day_of_week' , 'duration' , 'campaign' , 'pdays' , 'poutcome' , 'emp.var.rate' , 'cons.conf.idx' , 'cons.price.idx' , 'nr.employed','y')
testNew <- test_bank_20 %>% dplyr::select('job', 'education' , 'marital' , 'housing' , 'contact' , 'default' ,'month' , 'day_of_week' , 'duration' , 'campaign' , 'pdays' , 'poutcome' , 'emp.var.rate' , 'cons.conf.idx' , 'cons.price.idx' , 'nr.employed','y')

model <- svm(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm
################### Compare with Naive Bayes with Logistic Predictors############
model <- naiveBayes(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm



