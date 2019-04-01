# load libraries
library(forecast)
library(dplyr)
library(ggplot2)

main = function() {
  # this function consolidates all the steps to build Fourier Transform ARIMA model

  # load the data
  data = load_data()
  
  # split the data for train and test data. 
  # The dates are the start and end points for training data
  # test data are the one day after the training period
  dates = split_dates_for_timeseries_cv()
  
  # estimate how many Fourier terms are necessary by applying sliding window TS cross validation
  result = run_ts_sliding_window_cv(dates,data)

  # comparing Fourier terms by AICc and test RMSE. In other words, statsitical fit vs pridictive power.
  # If AICc's are close enough (less than 2% difference), focus on predictive power
  FourierTerms = analyze_result(result)
  
  # Once the best model is found,estimate (predict) the next hour (Aug.28, from 3pm to 4pm)
  predict_the_next_hour(FourierTerms)
}


load_data = function(myfile='../data/agg15data.csv') {
  # This function loads the data and transform the logintime to datetime object (POSIXct in R)
  
  data = read.csv(myfile,header=FALSE,col.names = c('login_time','count'))
  data$login_datetime = as.POSIXct(data$login_time)
  return(data)
}


split_dates_for_timeseries_cv = function() {
  # Use 3 month worth of data as one training set as there are enough data.
  # As the data ends at 2:45 pm on the 4th Saturday of August, 
  # traning date is timed to end on the same hour of the same week.
  # An exception is the last training set in which the end date is the 3rd Saturday
  # of August. This is because there needs to be a test set for this training set.
  # The only the uncontrolled remaning factor would be annual
  # but that is out of scope because the data provided has only 8 months
  
  dates = list()
  dates$start = c("2010-01-01 00:00:00","2010-03-27 15:00:00", "2010-06-26 15:00:00")
  dates$end = c("2010-03-27 14:45:00","2010-06-26 14:45:00","2010-08-21 14:45:00")
  return(dates)
}


run_ts_sliding_window_cv = function(dates,data) {
  # This function runs cross validation in order to estimate 
  # number of daily and weekly Fourier terms and ARIMA errors.
  
  # first set up a empty dataframe to store results of cross-validation
  result = data.frame(matrix(ncol=5,nrow=0))
  colnames(result) = c("FourierTerms96","FourierTerms672","AICc","TrainingPeriod","forecast")

  for (idx in 1:length(dates$start)) {
    # set up training data based on the date splits done earlier
    train_data = data[(data$login_datetime >= dates$start[idx]) & (data$login_datetime <= dates$end[idx]),]$count
    
    # produce AICc and forecast RMSE for each candidate model
    aic_vals = train_test_models(train_data,idx,6,6)
    
    # store the stats into result dataframe
    result = rbind(result,aic_vals)
  
  }

  return(result)
}


train_test_models = function(train_data,idx,FTerms96Max,FTerms672Max) {
  # This function produces AICc and forecast RMSE for candidate model,
  # each of which has daily (frequency 96) and weekly (frequency 672)
  # Fourier terms and ARIMA residuals
  
  aic_vals_temp = NULL
  aic_vals = NULL
  for (i in 1:FTerms96Max){
    for (j in 1:FTerms672Max){
      model = fit_fourier_with_arima(train_data,i,j)
      
      # forecast next hour and rmse of test data
      rmse = produce_forecast_rmse(model$fit,model$xreg,idx)
      
      aic_vals_temp = cbind(i,j,model$fit$aicc,idx,rmse)
      aic_vals = rbind(aic_vals,aic_vals_temp)
    }
  }
  colnames(aic_vals) = c("FourierTerms96","FourierTerms672","AICc","TrainingPeriod","forecast")
  aic_vals = data.frame(aic_vals)
  return(aic_vals)
}

fit_fourier_with_arima = function(train_data, FTerms96,FTerms672){
  # This function utilizes R forecast library to fit dynamic regression models 
  # with Fourier terms as exogenous regressors. The autocorrelated errors are
  # also taken care of with ARIMA technique.
  
  xreg1 = fourier(ts(train_data,frequency=96),K=FTerms96)
  xreg2 = fourier(ts(train_data,frequency=672),K=FTerms672)
  xregs = cbind(xreg1,xreg2)
  fit = auto.arima(train_data,seasonal=FALSE,xreg=xregs)
  return(list(fit=fit,xreg=xregs))
}
  
  
produce_forecast_rmse = function(fit,xregs,idx,h=96) {
  # Not only AICc but also RMSE of forecast could tip the scale for model selection 
  # as preidctive power is one the major criteria.
  # Here, we forecast for one day (96 periods) to produce RMSE 
  # to test the robustness of the model.
  
  # forecast based on the model
  mean_predictions = forecast(fit,xreg=xregs[1:h,])$mean
  # ground truth
  actual = data[(data$login_datetime > dates$end[idx]) & (data$login_datetime <= as.POSIXct(dates$end[idx]) + 60*(15*h)),]$count
  rmse = (mean((mean_predictions - actual)^2))^(1/2)
  return(rmse)
}

analyze_result = function(result) {
  # After AICc and RMSE are produced for all the candidate models,
  # the best model based on both of these criteria is chosen. 
  # However, if the differece in AICc is  negligible (ex. less than 2%),
  # then the best model is selected for the lowest RMSE.
  
  # get the best models on each of the metrics
  aic_lowest = result %>% group_by(TrainingPeriod) %>% summarize(AICc=min(AICc))
  rmse_lowest = result %>% group_by(TrainingPeriod) %>% summarize(forecast=min(forecast))
  
  # show AICc-based model
  aic_result = inner_join(result,aic_lowest,by=c("TrainingPeriod","AICc"))
  rmse_result = inner_join(result,rmse_lowest,by=c("TrainingPeriod","forecast"))
  print("result based on AICc")
  print(aic_result)
  
  # show RMSE-based model
  print("result based on RMSE")
  print(rmse_result)
  print("Comparing models'AICc difference in percentage for AICc-based and RMSE based approaches")
  print((rmse_result$AICc - aic_result$AICc)/aic_result$AICc*100)
  
  # looks like there is a noticable difference between training periods. 
  # As our goal is to estimate 1 hour of August 28 from 3pm, use the training period
  # closest to this time period.
  FourierTerms96 = rmse_result[rmse_result$TrainingPeriod == 3,]$FourierTerms96
  FourierTerms672 = rmse_result[rmse_result$TrainingPeriod == 3,]$FourierTerms672
  return(list(FourierTerms96 = FourierTerms96, FourierTerms672 = FourierTerms672))
}

predict_the_next_hour = function(FourierTerms) {
  # Once the best model is selected, fit the model again
  # and estimate 1 hour on August 28 from 3pm
  # In addition, check the residuals of ARIMA 
  # as the prediction interval is based on its distribution
  # estimated prediction variance.
  
  # fit the best model again to produce R object
  final_model = fit_fourier_with_arima(data[data$login_datetime >= dates$start[3],]$count,FourierTerms$FourierTerms96,FourierTerms$FourierTerms672)
  # print the 1 hr forecasts
  print(forecast(final_model$fit,xreg=final_model$xreg[1:4,])$mean)
  # plot the forecast along with 2 weeks worth of the past time series leading up to the estimates
  autoplot(forecast(final_model$fit,xreg=final_model$xreg[1:4,]),include=200)
  # check the time series plot, distribution and autocorrelation of residuals
  checkresiduals(final_model$fit,lag=100)
}

# Run the entire code
main()


