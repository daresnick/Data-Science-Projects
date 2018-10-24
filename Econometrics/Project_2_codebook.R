rm(list = ls())
gc()
par(mfrow=c(1,1))

library(ggplot2)
library(quantmod)
library(tseries)
library(forecast)
library(TSPred)
library(fAssets)
library(fPortfolio)
#library(fGarch)
library(timeSeries)
library(zoo)
library(PerformanceAnalytics)
library(dplyr)
library(rugarch)

call_stock <- function(x){
  prices = get.hist.quote(x, start = "2017-04-19", end = "2018-04-19", quote = "Close", compression = "d")
  #returns = CalculateReturns(prices, method="simple")
  df <- as.data.frame(prices)
  #df$date <- row.names(df)
  df$date <- time(prices)
  df <- df[,c(2,1)]
}

SPY <- call_stock("SPY")
NKE <- call_stock("NKE")
DIS <- call_stock("DIS")
KO <- call_stock("KO")
GS <- call_stock("GS")
JNJ <- call_stock("JNJ")
AMZN <- call_stock("AMZN")

SPY <- call_stock("SPY")
UA <- call_stock("UA")
DIS <- call_stock("DIS")
KO <- call_stock("KO")
BRKB <- call_stock("BRK-B")
EL <- call_stock("EL")
AMZN <- call_stock("AMZN")

colnames(SPY) <- c("Date", "SPY")
colnames(NKE) <- c("Date", "NKE")
colnames(DIS) <- c("Date", "DIS")
colnames(KO) <- c("Date", "KO")
colnames(GS) <- c("Date", "GS")
colnames(JNJ) <- c("Date", "JNJ")
colnames(AMZN) <- c("Date", "AMZN")

# Replace NKE -> UA, GS -> BRK-B, JNJ -> EL
UA <- call_stock("UA")
BRKB <- call_stock("BRK-B")
EL <- call_stock("EL")
colnames(UA) <- c("Date", "UA")
colnames(BRKB) <- c("Date", "BRK-B")
colnames(EL) <- c("Date", "EL")

##################################### END PULL DATA #####################################

##################################### START DATA EXPLORATION #####################################

IT <- Reduce(dplyr::inner_join,list(SPY, UA, DIS, KO, BRKB, EL, AMZN))
IT_test <- do.call(rbind,  lapply(split(IT,"Date"), function(w) last(w))) 

IT <- timeSeries(IT[, 2:8], IT[, 1])
#log(lag(IT) / IT)
IT_return <- returns(IT)
IT_return
chart.CumReturns(IT_return, legend.loc = 'topleft', main = '')

Spec = portfolioSpec()
setSolver(Spec) = "solveRshortExact"
Frontier <- portfolioFrontier(as.timeSeries(IT_return), Spec, constraints = "Short") #constraints could also ="Short"
frontierPlot(Frontier, col = rep('orange', 2), pch = 19)

#Split data into training and testing data set
data.train <- window(IT, start="2017-04-19", end="2018-03-01")
dim(as.matrix(data.train)) # 219
data.test <- window(IT, start="2018-03-02", end="2018-04-19")
dim(as.matrix(data.test)) # 33

# Split returns to test train
IT_return.ts <- ts(IT_return)
#IT_return.ts.test <- IT_return.ts[247:251,]
data_return.train <- window(IT_return.ts, start=1, end=246)
dim(as.matrix(data.train)) # 219
data_return.test <- window(IT_return.ts, start=247, end=251)
dim(as.matrix(data.test)) # 33

##################################### END DATA EXPLORATION #####################################

##################################### START ARIMA #####################################
plot(IT_return)
plot(IT_return[,'KO'])
plot(data_return.train[,'KO'])
plot(data_return.test[,'KO'])


# Get best variations
#bestmodel_KO <- auto.arima(data_return.train[,'KO'], trace=TRUE, test="pp", ic="bic")
bestmodel_KO <- arima(data_return.train[,'KO'], c(2,0,2))
summary(bestmodel_KO)
confint(bestmodel_KO)
tsdiag(bestmodel_KO)

# Forecast data
bestmodel_KO.forecast <- forecast(bestmodel_KO, h=100)
bestmodel_KO.forecast
plot(bestmodel_KO.forecast[["residuals"]], xlab="Day", ylab="Residuals")
plot(bestmodel_KO.forecast, xlab="Day", ylab="Returns")


# Plot close up
plotarimapred(data_return.test[,'KO'], bestmodel_KO, xlim=c(100, 200), range.percent = 0.05)
accuracy(bestmodel_KO.forecast, data_return.test[,"KO"])
plot(bestmodel_KO)
accuracy(bestmodel_KO.forecast)
# TODO: Rest of ARIMAS

##################################### END ARIMA #####################################

##################################### START ARCH #####################################


# Set to GARCH(1,1)
IT_garch11_spec <- ugarchspec(variance.model = list(
  garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)))

IT_garch11_fit <- ugarchfit(spec = IT_garch11_spec, data = data_return.train[,'KO'])
IT_garch11_fit

# Forecast using GARCH
IT_garch11_fcst <- ugarchforecast(IT_garch11_fit, n.ahead = 5)
garch.ts <-ts(IT_garch11_fcst@forecast$seriesFor, start=247, end=251)

accuracy(garch.ts, data_return.test[,'KO'])


# Backtesting
IT_garch11_roll <- ugarchroll(IT_garch11_spec,  data_return.train[,'KO'],
  n.start = 10, refit.every = 1, refit.window = "moving",
  solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.05,
  keep.coef = TRUE, solver.control=list(tol=1e-6, trace=1), fit.control=list(scale=1))
#warnings()
# Try to resume - not working
IT_garch11_roll = resume(IT_garch11_roll, solver="gosolnp")

report(IT_garch11_roll, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)
# If the return is more negative than the VaR, we have a VaR exceedance. In our case, a VaR exceedance should only occur in 5% of the cases (since we speci ed a 95% con dence level).
IT_VaR <- zoo(IT_garch11_roll@forecast$VaR[, 1])
index(IT_VaR) <- as.yearmon(rownames(IT_garch11_roll@forecast$VaR))
IT_actual <- zoo(IT_garch11_roll@forecast$VaR[, 2])
index(IT_actual) <-
  as.yearmon(rownames(IT_garch11_roll@forecast$VaR))

plot(IT_actual, type = "b", main = "95% VaR Backtesting",
   xlab = "Date", ylab = "Return/VaR in percent")
lines(IT_VaR, col = "red")
legend("bottomleft", inset=.05, c("IT return","VaR"), col = c("black","red"), lty = c(1,1))



##################################### END ARCH #####################################

##################################### START OPTIMIZATIONS #####################################

## To solve for a return...
#??portfolioSpec
Spec <- portfolioSpec()
setSolver(Spec) <- "solveRshortExact" #Set the method for solving...See documentation for calculation options...
#solveRshortExact allows for unlimited short selling..."solveRquadprog" can be used for not short selling
setTargetReturn(Spec) <- mean(colMeans(IT_return)) # to set target at average returns of all columns
#setTargetReturn(Spec) <- 0.08/52 # for weekly data

efficientPortfolio(IT_return, Spec, 'Short') #Could set for LongOnly...Need to make sure spec is not solveRshortExact or will override...

minvariancePortfolio(IT_return, Spec, 'Short')
minriskPortfolio(IT_return, Spec)
maxreturnPortfolio(IT_return, Spec)

tangencyPortfolio(IT_return, Spec, 'Short') 

#highest return/risk ratio on the efficient frontier
#For the MarKOwitz portfolio this is the same as the Sharpae ratio. To find this point on the 
#frontier the return/risk ratio calculated from the target return and target risk returned 
#by the function efficientPortfolio. Note, the default value of the risk free rate is zero. 

#now let's have some fun plotting this...
#?frontierPlot #your available plots...

#Lets get the frontier in a little different way...
frontier=portfolioFrontier(as.timeSeries(IT_return))
frontierPlot(frontier)
grid()



tailoredFrontierPlot(frontier,
                     return = c("mean", "mu"), risk = c("Cov", "Sigma", "CVaR", "VaR"),
                     mText = NULL, col = NULL, ylim = NULL, 
                     twoAssets = FALSE, sharpeRatio = FALSE, title = TRUE,
                     xlim = c(0.005,0.02))


frontier-plot(frontier,
              return = c("mean", "mu"), risk = c("Cov", "Sigma", "CVaR", "VaR"),
              mText = NULL, col = NULL, xlim = NULL, ylim = NULL,
              twoAssets = FALSE, sharpeRatio = TRUE, title = TRUE)

weightsPlot(frontier) #Black line is minimum variance portfolio

#Tangency Portfolio Graphs
tgPort=tangencyPortfolio(IT_return)
weightsPie(tgPort) #weights of securities in tangency portfolio

weightedReturnsPie(tgPort) #pie chart of weighted returns of the tangency portfolio

# Remove NKE, GS, JNJ
IT2 <- Reduce(dplyr::inner_join,list(SPY, DIS, KO, AMZN))
IT2_test <- do.call(rbind,  lapply(split(IT2,"Date"), function(w) last(w))) 
IT2 <- timeSeries(IT2[, 2:5], IT2[, 1])
log(lag(IT2) / IT2)
IT2_return <- returns(IT2)
IT2_return

# Replace NKE -> UA -> NFLX, GS -> BRK-B, JNJ -> EL -> FB
NFLX <- call_stock("NFLX")
BRKB <- call_stock("BRK-B")
FB <- call_stock("FB")

colnames(NFLX) <- c("Date", "NFLX")
colnames(BRKB) <- c("Date", "BRK-B")
colnames(FB) <- c("Date", "FB")

IT3 <- Reduce(dplyr::inner_join,list(SPY, DIS, KO, AMZN, UA, BRKB, EL))
IT3_test <- do.call(rbind,  lapply(split(IT3,"Date"), function(w) last(w))) 
IT3 <- timeSeries(IT3[, 2:8], IT3[, 1])
log(lag(IT3) / IT3)
IT3_return <- returns(IT3)
IT3_return

Spec = portfolioSpec()
setSolver(Spec) = "solveRshortExact"
Frontier <- portfolioFrontier(as.timeSeries(IT_return), Spec, constraints = "Short") #constraints could also ="Short"
Frontier2 <- portfolioFrontier(as.timeSeries(IT2_return), Spec, constraints = "Short")
Frontier3 <- portfolioFrontier(as.timeSeries(IT3_return), Spec, constraints = "Short")
frontierPlot(Frontier, col = rep('orange', 2), pch = 19)
frontierPlot(Frontier2, col = rep('green',2), pch = 19, add = TRUE)
frontierPlot(Frontier3, col = rep('blue',2), pch = 19, add = TRUE)

##################################### END OPTIMIZATIONS #####################################


# Set to GARCH(1,1)
IT_garch11_spec <- ugarchspec(variance.model = list(
  garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)))

custom_garch <- function(spec, df, column){

IT_garch11_fit <- ugarchfit(spec = spec, data = df[,column])

# Forecast using GARCH
IT_garch11_fcst <- ugarchforecast(IT_garch11_fit, n.ahead = 12)



# Backtesting
IT_garch11_roll <- ugarchroll(spec,  df[,column],
                              n.start = 10, refit.every = 1, refit.window = "moving",
                              solver = "hybrid", calculate.VaR = TRUE, VaR.alpha = 0.05,
                              keep.coef = TRUE, solver.control=list(tol=1e-6, trace=1), fit.control=list(scale=1))
#warnings()
# Try to resume - not working
IT_garch11_roll = resume(IT_garch11_roll, solver="gosolnp")

report <-report(IT_garch11_roll, type = "VaR", VaR.alpha = 0.05, conf.level = 0.95)

# If the return is more negative than the VaR, we have a VaR exceedance. In our case, a VaR exceedance should only occur in 5% of the cases (since we speci ed a 95% con dence level).
IT_VaR <- zoo(IT_garch11_roll@forecast$VaR[, 1])
index(IT_VaR) <- as.yearmon(rownames(IT_garch11_roll@forecast$VaR))
IT_actual <- zoo(IT_garch11_roll@forecast$VaR[, 2])
index(IT_actual) <-
  as.yearmon(rownames(IT_garch11_roll@forecast$VaR))

#plot(IT_actual, type = "b", main = "95% VaR Backtesting",
#     xlab = "Date", ylab = "Return/VaR in percent")
#lines(IT_VaR, col = "red")
#legend("topright", inset=.05, c("IT return","VaR"), col = c("black","red"), lty = c(1,1))
list(report =report, IT_VAR = IT_VaR, IT_actual =IT_actual)
}

#KO <- call_stock("KO")
#KO <- call_stock("KO")
#KO <- call_stock("KO")
#CO <- call_stock("CO")
#GS <- call_stock("GS")
#JNJ <- call_stock("JNJ")
#KO <- call_stock("KO")
#joe_garch <- custom_garch(IT_garch11_spec,data_return.train,'KO')
stocks <- c("SPY","NKE","DIS","GS","JNJ","KO","AMZN")
stock_garch <- list()

for(i in 1:length(stocks)){
  stock_garch[[i]] <- custom_garch(IT_garch11_spec,data_return.train,stocks[i])
}


SPY <- call_stock("SPY")
UA <- call_stock("UA")
DIS <- call_stock("DIS")
KO <- call_stock("KO")
BRKB <- call_stock("BRK-B")
EL <- call_stock("EL")
AMZN <- call_stock("AMZN")


stocks <- c("SPY","UA","DIS","BRK-B","EL","KO","AMZN")


garch_accuracy <- function(x){
IT_garch11_spec <- ugarchspec(variance.model = list(
  garchOrder = c(1, 1)),
  mean.model = list(armaOrder = c(0, 0)))

IT_garch11_fit <- ugarchfit(spec = IT_garch11_spec, data = data_return.train[,x])
IT_garch11_fit

# Forecast using GARCH
IT_garch11_fcst <- ugarchforecast(IT_garch11_fit, n.ahead = 5)
garch.ts <-ts(IT_garch11_fcst@forecast$seriesFor, start=247, end=251)

print(accuracy(garch.ts, data_return.test[,x]))
}


for (i in stocks){
  garch_accuracy(i)
  print(i)
}

arima_accuracy <- function(x){
  bestmodel_KO <- arima(data_return.train[,x], c(2,0,2))
  
  # Forecast data
  bestmodel_KO.forecast <- forecast(bestmodel_KO, h=100)
  
  
  # Plot close up
  accuracy(bestmodel_KO.forecast)
}



for (i in stocks){
  print(arima_accuracy(i))
  print(i)
}





