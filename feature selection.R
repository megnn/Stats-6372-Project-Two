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

############# Compare with KNN ########################
library(class)
balancetrainNew <- balanced_train_bank_20 %>% dplyr::select('job', 'default', 'contact', 'duration', 'month', 'poutcome', 'campaign','y')
testNew <- test_bank_20 %>% dplyr::select('job', 'default', 'contact', 'duration', 'month', 'poutcome', 'campaign','y')
classifications = knn(balanced_train_bank_20[,c('job', 'default', 'contact', 'duration', 'month', 'poutcome', 'campaign')],test_bank_20[,c('job', 'default', 'contact', 'duration', 'month', 'poutcome', 'campaign')],as.factor(balanced_train_bank_20$y), prob = TRUE, k = 5)
table(test_bank_20$y,classifications)
cfm = confusionMatrix(classifications,test_bank_20$y)
cfm

################ Compare with SVM ###############
library(e1071)
model <- svm(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm

################### Compare with Naive Bayes ############
model <- naiveBayes(y~.,data = balancetrainNew)
predict = predict(model, testNew)

#Confusion Matrix
cfm = confusionMatrix(predict,testNew$y)
cfm

