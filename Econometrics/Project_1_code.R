
## Code for Project 1
## Damon Resnick, Kaitlin Kirasich, Joe Cook


library(quantmod)
library(tseries)
library(forecast)
library(tspred)

library(quantmod)
library(tseries)
library(forecast)

stocks <- c("NFLX", "NVDA", "FB", "SQ", "SHOP", "RHT", "Z", "TXN", "MTCH")
start_date <- "2016-01-01"


for(i in 1:length(stocks)){
  data.frame(getSymbols(stocks[i],from=start_date))
}

a<- merge(FB, MTCH, by=0, all = TRUE)
a <- merge(a, NFLX, by=0, all = TRUE)
a <- merge(a, NVDA, by=0, all = TRUE)
a <- merge(a, RHT, by=0, all = TRUE)
a <- merge(a, SHOP, by=0, all = TRUE)
a <- merge(a, SQ, by=0, all = TRUE)
a <- merge(a, TXN, by=0, all = TRUE)
a <- merge(a, Z, by=0, all = TRUE)

a <- data.frame(a)

b <- grep("by",colnames(a))

a <- a[,-b]

a.ts <- ts(a)


reg_func <- function(df)
{
  check <- as.numeric(df[!is.na(df)])
  #regdata <- data.frame(check = check[3:length(check)], index = 1:(length(check)-2), prev_day = check[1:(length(check)-2)], two_day = check[2:(length(check)-1)])
  regdata <- data.frame(check = check[1:(length(check))], index = 1:(length(check)))
  regdata_train <- regdata[1:floor(length(check)-10),]
  regdata_test <- regdata[floor(length(check)-9):nrow(regdata),]
  reg <- lm(check ~ index, regdata_train)
  summary(reg)
  plot(regdata_train$check)
  lines(predict(reg), col = "red")
  
  
  
  predicted <- c(predict(reg),predict(reg, regdata_test))
  plot(regdata$check)
  lines(predicted, col = "red")
  list(rmse = accuracy(predict(reg, regdata_test),regdata_test$check)[2],predicted = predicted)
}

ghs <- reg_func(a$NFLX.Close)
stocks_predict <- list(a$NFLX.Close, a$NVDA.Close, a$FB.Close, a$SQ.Close, a$SHOP.Close, a$RHT.Close, a$Z.Close, a$TXN.Close, a$MTCH.Close)

accuracy_scores <- list()
for(i in 1:length(stocks_predict)){
  accuracy_scores[[i]] <- reg_func(stocks_predict[[i]])[1]
}
#accuracy(predict(reg, regdata_test),regdata_test$check)[2]

rmse_scores <- unlist(accuracy_scores)
names(rmse_scores) <- paste(stocks,"rmse", sep="_")
rmse_scores <- data.frame(rmse_scores)


Predicted_scores <- list()
for(i in 1:length(stocks_predict)){
  Predicted_scores[[i]] <- reg_func(stocks_predict[[i]])[2]
}
Predicted_scores_df <- do.call(data.frame, Predicted_scores)
names(Predicted_scores_df) <- paste(stocks,"Reg_Predict", sep="_")

data.train <- window(a.ts[,'NFLX.Close'], start=1, end=554)
plot(data.train)
dim(as.matrix(data.train))
data.test <- window(a.ts[,'NFLX.Close'], start=555, end=564)
plot(data.test)
dim(as.matrix(data.test))

bestmodel1b <- auto.arima(data.train, seasonal = TRUE, stepwise = FALSE, approximation=FALSE, trace=TRUE, test="kpss", ic="aic")
bestmodel1b

summary(bestmodel1b)
confint(bestmodel1b)
tsdiag(bestmodel1b)

bestmodel1b.forecast <- forecast(bestmodel1b, h=10)
bestmodel1b.forecast
plot(bestmodel1b.forecast, xlab="Day", ylab="Closing Price", xlim=c(1, 564))

plotarimapred(data.test, bestmodel1b, xlim=c(450, 564), range.percent = 0.05)
accuracy(bestmodel1b.forecast, data.test)
plotarimapred(data.test, bestmodel1b.forecast, xlim=c(548, 564), range.percent = 0.05)


arima_predictions <- function(x){
  #Split data into training and testing data set
  data.train <- window(a.ts[,x], start=1, end=554)
  plot(data.train)
  dim(as.matrix(data.train))
  data.test <- window(a.ts[,x], start=555, end=564)
  plot(data.test)
  dim(as.matrix(data.test))
  
  bestmodel1b <- auto.arima(data.train, seasonal = TRUE, stepwise = FALSE, approximation=FALSE, trace=TRUE, test="kpss", ic="aic")
  
  bestmodel1b.forecast <- forecast(bestmodel1b, h=10)
  predicted <- c(bestmodel1b$fitted,bestmodel1b.forecast$mean)
  predicted
}

stocks_predict_ar <- list("NFLX.Close", "NVDA.Close", "FB.Close", "SQ.Close", "SHOP.Close", "RHT.Close", "Z.Close", "TXN.Close", "MTCH.Close")

arima_list <- list()
for(i in 1:length(stocks_predict_ar)){
  arima_list[[i]] <- arima_predictions(stocks_predict_ar[[i]])
  print(i)
}
Predicted_scores_ar <- do.call(data.frame, arima_list)
names(Predicted_scores_ar) <- paste(stocks_predict_ar,"AR_Predict", sep="_")
b <- grep("Close",colnames(a))
slimdf <- a[,b]
everything <- data.frame(slimdf, Predicted_scores_df, Predicted_scores_ar)
write.csv(everything,"All_close_predictions.csv")
