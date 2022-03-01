##############################################
####    Forecast cherry bloom date    ########
####    Miaoshiqi Liu and Siyue Yang  ########
####           February 28, 2022      ########
##############################################

library(fpp2)       # Fit time series.
library(rnoaa)      # Get weather data.
library(tidyverse)
library(ggplot2) 
library(imputeTS)   # Imputation on the temperature
library(rnoaa) 
library(lubridate)  # Date data. 

#-----------------------------------------------#
#                  Weather data                 #
#-----------------------------------------------#

# Get the tmax, prcp, tmin
get_ghcnd <- function(station_id = "JA000047759",
                      date_range = c("1950-01-01", "2022-02-28")) {
  
  weather <- ghcnd_search(stationid = station_id, var = c("tmax"), 
                        date_min = date_range[1], date_max = date_range[2])[[1]]

  for (info in c("tmin", "prcp")) {
    info_tmp <- ghcnd_search(stationid = station_id, var = c(info), 
                             date_min = date_range[1], date_max = date_range[2])[[1]]
    weather <- merge(weather, info_tmp, by = "date") 
  }
  
  weather <- weather %>% select(date, tmax, tmin, prcp) %>%
    mutate(tmax = tmax/10, tmin = tmin/10) %>%   # tenth of the C
    mutate(year = as.integer(format(date, "%Y")),
           month = as.integer(strftime(date, '%m')) %% 12, # make December "0"
           season = cut(month, breaks = c(0, 2, 5, 8, 11),
                        include.lowest = TRUE,
                        labels = c("Winter", "Spring", "Summer", "Fall")),
           year = if_else(month == 0, year + 1L, year)) %>%
    mutate(month_name = if_else(month == 0, 12, month)) %>%
    mutate(month_name = factor(month.name[as.integer(month_name)], levels = month.name)) 
  
  return(weather)
}

# Impute temperature data. 
imp_temperature <- function(dat = weather, temps = "tmin") { 
  
  for (temp in temps) {
    # Convert to time series
    temp_ori <- dat[, temp]
    temp_series <- ts(temp_ori, frequency = 1)
    dat$na <- is.na(temp_ori)
    
    # Use Kalman smoother to impute. Other options are also available.
    dat$imp <- na_kalman(temp_series)
    
    n_col <- ncol(dat)
    colnames(dat)[n_col-1] <- paste0(temp, "_na")
    colnames(dat)[n_col] <- paste0(temp, "_imp")
  }
  return(dat)
}

# Accumulative freezing days. 
get_afdd <- function(dat) {
  
  freeze_days <- dat %>% mutate(yday = yday(date)) %>% group_by(year) %>% filter(tmin_imp <= 0)
  
  # Number of accumulative freezing days
  freeze_days_n <- freeze_days %>% group_by(year) %>% summarise(afdd = n()) 
  
  # First and last freeze
  first_freeze <- freeze_days %>% group_by(year) %>% summarise(first_freeze_date = min(date))
  last_freeze <- freeze_days %>% group_by(year) %>% summarise(last_freeze_date = max(date))
  
  first_freeze_yday <- first_freeze %>% mutate(first_freeze_yday = yday(first_freeze_date))
  last_freeze_yday <- last_freeze %>% mutate(last_freeze_yday = yday(last_freeze_date))      
  
  year_range <- c(min(dat$year) : max(dat$year))
  
  freeze_info <- merge(data.frame(year = year_range), freeze_days_n, all.x = TRUE) 
  freeze_info <- merge(freeze_info, first_freeze_yday, by = "year", all.x = TRUE)
  freeze_info <- merge(freeze_info, last_freeze_yday, by = "year", all.x = TRUE)
  
  return(freeze_info)
}

# Accumulative growing days. 
get_agdd <- function(dat) {
  
  growing_days <- dat %>% mutate(yday = yday(date)) %>% group_by(year) %>% 
    mutate(tavg_cal = (tmax_imp + tmin_imp)/2) %>% filter(tavg_cal > 0)
  
  # Number of accumulative growing days
  grow_days_n <- growing_days %>% group_by(year) %>% summarise(agdd = n()) 
  
  # First and last freeze
  first_grow <- growing_days %>% group_by(year) %>% summarise(first_grow_date = min(date))
  last_grow <- growing_days %>% group_by(year) %>% summarise(last_grow_date = max(date))
  
  first_grow_yday <- first_grow %>% mutate(first_grow_yday = yday(first_grow_date))
  last_grow_yday <- last_grow %>% mutate(last_grow_yday = yday(last_grow_date))      
  
  year_range <- c(min(dat$year) : max(dat$year))
  
  freeze_info <- merge(data.frame(year = year_range), grow_days_n, all.x = TRUE) 
  freeze_info <- merge(freeze_info, first_grow_yday, by = "year", all.x = TRUE)
  freeze_info <- merge(freeze_info, last_grow_yday, by = "year", all.x = TRUE)
  
  return(freeze_info)
}

# Generate weather variables for the model fitting.  
get_seasons.temperature <- function(station_id, date_range = c("1950-01-01", "2022-01-31")){
  
  # Get weather data. 
  dailyweather = get_ghcnd(station_id = station_id, date_range = date_range)
  
  # Impute. 
  dailyweather = imp_temperature(dat = dailyweather, c("tmax", "tmin"))
  
  agdd <- get_agdd(dailyweather)
  afdd <- get_afdd(dailyweather)
  
  final_dat <- merge(agdd, afdd, by = "year")
  
  temp_prcp <- dailyweather %>%
    group_by(year, season) %>%
    summarize(tmax_avg = mean(tmax_imp), 
              tmin_avg = mean(tmin_imp), 
              prcp_avg = mean(prcp, na.rm = TRUE)) %>%
    filter(season %in% c("Spring", "Winter")) %>% 
    pivot_wider(names_from = "season", values_from = c("tmax_avg", "tmin_avg", "prcp_avg"))
  
  final_dat <- merge(final_dat, temp_prcp, by = "year")

  return(final_dat)
}


#-----------------------------------------------#
#                 Local linear model            #
#-----------------------------------------------#

epak_fun <- function(tcenter, t, bw){
  n = length(t)
  k = rep(0, n)
  for(i in 1:n){
    k[i] = single_epak_fun((t[i] - tcenter)/bw)
  }
  return(k) 
}

quadratic_single_fun <- function(teval, t_train, Y, design_matrix, bw){
  p = NCOL(design_matrix)
  X = design_matrix
  tcenter = teval
  wt = epak_fun(tcenter, t_train, bw)
  X_append = X*(t_train - tcenter)
  X_new = cbind(X, X_append)
  traindata = cbind(Y, X_new)
  traindata = data.frame(traindata)
  est = lm(Y~.+0, data = traindata, weights = wt)$coef
  beta_hat = as.matrix(est[1:p])
  return(beta_hat)
}

Three_Cities_kernel_fun <- function(loc, final.dat, bw){
  dat = final.dat %>%
    filter(location == loc) 
  
  dat = na.omit(dat)
  
  train.dat = dat[1:(NROW(dat) - 10), ]
  test.dat = dat[(NROW(dat) - 9): NROW(dat), ]
  
  N = NROW(train.dat)
  t = (1:N)/N
  Y = train.dat$bloom_doy
  X = cbind(rep(1, N), train.dat$afdd, train.dat$tmin_avg_Spring, train.dat$tmax_avg_Spring)
  colnames(X) <- c("Intercept", "afdd", "tmin_Spring", "tmax_Spring")
  p = NCOL(X)
  
  beta_hat = lapply(t, quadratic_single_fun, t_train = t, Y = Y, design_matrix = X, bw = bw)
  
  estimated.intercept = Reduce('rbind', lapply(beta_hat, FUN = function(x){return(x['Intercept',])}))
  estimated.afdd = Reduce('rbind',lapply(beta_hat, FUN = function(x){return(x['afdd',])}))
  estimated.tmin_Spring = Reduce('rbind',lapply(beta_hat, FUN = function(x){return(x['tmin_Spring',])}))
  estimated.tmax_Spring = Reduce('rbind',lapply(beta_hat, FUN = function(x){return(x['tmax_Spring',])}))
  
  local.linear.fitted.values = estimated.intercept + estimated.afdd*train.dat$afdd + estimated.tmax_Spring*train.dat$tmax_avg_Spring + estimated.tmin_Spring*train.dat$tmin_avg_Spring
  
  local.linear.fitted.values = round(local.linear.fitted.values, 0)
  R_square = 1 - sum((Y - local.linear.fitted.values)^2)/sum((Y - mean(Y))^2)
  
  coef.int = mean(estimated.intercept[(N - 3*floor(bw*N)):(N - floor(bw*N))])
  coef.afdd = mean(estimated.afdd[(N - 3*floor(bw*N)):(N - floor(bw*N))])
  coef.tmin_Spring = mean(estimated.tmin_Spring[(N - 3*floor(bw*N)):(N - floor(bw*N))])
  coef.tmax_Spring = mean(estimated.tmax_Spring[(N - 3*floor(bw*N)):(N - floor(bw*N))])
  
  
  test.fit = coef.int + coef.afdd*test.dat$afdd + coef.tmin_Spring*test.dat$tmin_avg_Spring +   coef.tmax_Spring*test.dat$tmax_avg_Spring
  test.fit = round(test.fit, 0)
  
  train.MAE = mean(abs(Y - local.linear.fitted.values))
  test.MAE = mean(abs(test.dat$bloom_doy - test.fit))
  output = list(loc, R_square, train.MAE, test.MAE, test.fit)
  names(output) <- c("location", "R_square", "train.MAE", "test.MAE", "test fit")
  return(output)
}

single_epak_fun <- function(x){
  if(abs(x) <= 1){
    value = (3/4)*(1-x^2)
  }
  else{value = 0}
  return(value)
}

#-----------------------------------------------#
#                 Forecast                      #
#-----------------------------------------------#

# Fit time series. 
fit_ts <- function(ts_weather, freq = 12) {
  # Time sereies plots
  # autoplot(ts_weather)
  # ggseasonplot(ts_weather) 
  # ggsubseriesplot(ts_weather)
  
  # Fit arima model
  fit_arima <- auto.arima(ts_weather)  
  #coef_arima <- print(summary(fit_arima))
  #res_arima <- checkresiduals(fit_arima)
  
  # Forecast with arima fitted model
  fcst <- forecast(fit_arima, h = 10*freq, simulate = TRUE)
  #autoplot(fcst, include = 100)
  
  # Transform the forecasting data into the same form of the input data
  fcst$time <- row.names(fcst)
  fcst <- round(fcst$mean, 2)
  
  if (freq == 12) {
    year_name <- rep(2022:2031, each = 12)
    period_name <- rep(1:12, 10)
    
    forecast_weather <- data.frame(year_name, period_name, fcst) %>%
      mutate(month = case_when(period_name == 12 ~ 0, 
                               TRUE ~ as.numeric(period_name)), # make December "0"
             year = if_else(month == 0, year_name + 1L, year_name)) %>% 
      select(year, month, fcst)
    
  } else if (freq == 1) {
    year <- rep(2022:2031, each = 1)
    forecast_weather <- data.frame(year, fcst) 
  } else {
    year <- rep(2022:2031, each = 4)
    season <- rep(c("Winter", "Spring", "Summer", "Fall"), 10)
    forecast_weather <- data.frame(year, season, fcst) 
  }
  
  return(forecast_weather)
}

# Forecast the next 10 years weather. 
forecast_weather <- function(dailyweather) {
  
  # Get afdd 
  afdd <- get_afdd(dailyweather) %>% select(year, afdd)
  
  # Summarise by season and year
  dailyweather <- dailyweather %>%
    group_by(year, season) %>%
    summarise(tmax_avg = mean(tmax_imp),
              tmin_avg = mean(tmin_imp),
              prcp_avg = mean(prcp, na.rm = TRUE)) 
  
  # 1. Avgerage max
  ts_weather_max <- dailyweather %>%
    select(year, season, tmax_avg) %>%
    pivot_wider(names_from = c("year", "season"), values_from = "tmax_avg") %>%
    transpose()
  
  ts_weather_max <- ts(ts_weather_max, start = c(1951, 1), frequency = 4)
  forecast_weather_tmax <- fit_ts(ts_weather_max, freq = 4)
  colnames(forecast_weather_tmax) <- c("year", "season", "tmax_avg")
  
  # 2. Average min 
  ts_weather_min <- dailyweather %>%
    select(year, season, tmin_avg) %>%
    pivot_wider(names_from = c("year", "season"), values_from = "tmin_avg") %>%
    transpose()
  
  ts_weather_min <- ts(ts_weather_min, start = c(1951, 1), frequency = 4)
  forecast_weather_tmin <- fit_ts(ts_weather_min, freq = 4)
  colnames(forecast_weather_tmin) <- c("year", "season", "tmin_avg")
  
  # 3. Avgerage prcp
  ts_weather_prcp <- dailyweather %>%
    select(year, season, prcp_avg) %>%
    pivot_wider(names_from = c("year", "season"), values_from = "prcp_avg") %>%
    transpose()
  
  ts_weather_prcp <- ts(ts_weather_prcp, start = c(1951, 1), frequency = 4)
  forecast_weather_prcp <- fit_ts(ts_weather_prcp, freq = 4)
  colnames(forecast_weather_prcp) <- c("year", "season", "prcp_avg")
  
  # 4. afdd prcp
  ts_weather_afdd <- ts(afdd[, 2], start = c(1951, 1), frequency = 1)
  # Impute missing value
  ts_weather_afdd <- na_kalman(ts_weather_afdd)
  forecast_weather_afdd <- fit_ts(ts_weather_afdd, freq = 1)
  colnames(forecast_weather_afdd) <- c("year", "afdd")
  
  forecast_weather <- cbind(forecast_weather_tmax, 
                            tmin_avg = forecast_weather_tmin$tmin_avg, 
                            prcp_avg = forecast_weather_prcp$prcp_avg) 
  
  # Join with the Dec data from the original 
  # to get the winter season summary
  # forecast_weather <- dailyweather %>%
  #   filter(month == 0 & year == 2022) %>%
  #   bind_rows(forecast_weather) %>%
  #   mutate(season = cut(month, breaks = c(0, 2, 5, 8, 11),
  #                       include.lowest = TRUE,
  #                       labels = c("Winter", "Spring", "Summer", "Fall"))) %>%
  #   group_by(year, season) %>%
  #   summarize(tmax_avg = mean(tmax_avg, na.rm = T), 
  #             tmin_avg = mean(tmin_avg, na.rm = T), 
  #             prcp_avg = mean(prcp_avg, na.rm = T)) %>%
  #   filter(season %in% c("Spring", "Winter")) %>% 
  #   pivot_wider(names_from = "season", 
  #               values_from = c("tmax_avg", "tmin_avg", "prcp_avg")) %>%
  #   filter(year < 2032) 
  
  forecast_weather <- forecast_weather %>%
    filter(season %in% c("Spring", "Winter")) %>% 
    pivot_wider(names_from = "season", 
                values_from = c("tmax_avg", "tmin_avg", "prcp_avg")) %>%
    filter(year < 2032) 
  
  forecast_weather <- merge(forecast_weather, forecast_weather_afdd, by = "year")
  
  return(forecast_weather)
}

forecast_single <- function(var) {
  ts_weather <- ts(var, start = c(1951, 1), frequency = 1)
  forecast_weather <- fit_ts(ts_weather, freq = 1)
  return(forecast_weather)
}

# Forecast the next 10 years weather. 
forecast_weather_new <- function(weather_data, 
                                 forecast_var = c("tmax_avg_Winter", 
                                                  "tmax_avg_Spring", 
                                                  "tmin_avg_Winter", 
                                                  "tmin_avg_Spring",
                                                  "prcp_avg_Winter", 
                                                  "prcp_avg_Spring",
                                                  "afdd")) {
  
  forecast_weather <- forecast_single(weather_data$tmax_avg_Winter)
  colnames(forecast_weather) <- c("year", "tmax_avg_Winter")
  forecast_var <- forecast_var[-1]
  
  for (var in forecast_var) {
    weather1 <- forecast_single(weather_data[, var])
    colnames(weather1) <- c("year", paste0(var))
    forecast_weather <- merge(forecast_weather, weather1, by = "year")
  }
  
  return(forecast_weather)
}



#-----------------------------------------------#
#                 Calculate error               #
#-----------------------------------------------#

cal_pred <- function(train_mod, train_data, test_data) {
  
  y_hat_train <- predict(train_mod, data = train_data)
  y_hat_test <- predict(train_mod, newdata = test_data)
  
  test_MSE <- mean(abs(test_data$bloom_doy - round(y_hat_test)))
  
  train_pred <- round(y_hat_train)
  test_pred <- round(y_hat_test)
  
  train_mse <- mean((train_data$bloom_doy - train_pred)^2)
  test_mse <- mean((test_data$bloom_doy - test_pred)^2)
  
  train_mae <- mean(abs(train_data$bloom_doy - train_pred))
  test_mae <- mean(abs(test_data$bloom_doy - test_pred))

  nume <- sum((train_data$bloom_doy - y_hat_train)^2)
  meany <- mean(train_data$bloom_doy)
  deno <- sum((train_data$bloom_doy - rep(meany, length(y_hat_train)))^2)
  R_sq <- 1 - nume/deno
  
  return(list(train_pred = train_pred, test_pred = test_pred,
              train_mae = train_mae, test_mae = test_mae,
              train_mse = train_mse, test_mse = test_mse,
              R_sq = 1 - nume/deno))
}
