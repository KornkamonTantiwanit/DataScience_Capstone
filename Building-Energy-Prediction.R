library(readr)
library(knitr)
library(dplyr)
library(ggplot2)
library(lubridate)
library(hms)
library(rsample)
library(lightgbm)
library(randomForest)
library(caret)

#####################################################################################################
# Download dataset from Github Repository 
options(timeout = 120)

base_url <- "https://github.com/KornkamonTantiwanit/DataScience_Capstone/raw/main/"
files <- c("weather_hourly.zip", "2018Floor2.zip", "2019Floor2.zip")
urls <- paste0(base_url, files)
names(urls) <- files

all_downloads_successful <- TRUE

for (file_name in names(urls)) {
  tryCatch({
    download.file(urls[[file_name]], destfile = file_name, method = "auto") # Download file
    if (grepl("\\.zip$", file_name)) {
      unzip(file_name, exdir = ".") # Unzip into the current directory
      file.remove(file_name)} # Remove the zip file after unzipping
  }, error = function(e) {
    all_downloads_successful <<- FALSE})}

if (all_downloads_successful) cat("Download complete!") else cat("Some downloads failed.")

#####################################################################################################    
# Understand dataset 
# Weather data (Features)
weather_2018_2019 <- read_csv("weather_hourly.csv", show_col_types = FALSE) %>%
                     mutate(time = as.POSIXct(time))
kable(head(weather_2018_2019), format = "markdown", align = "l")

# Weahter Data Visualization 
ggplot(weather_2018_2019, aes(x = time)) +
  geom_line(aes(y = rhum, color = "Relative Humidity (RH)"), linewidth = 0.25) +  # RH line first
  geom_line(aes(y = temp, color = "Temperature (C)"), linewidth = 0.25) +  # Temperature line second
  geom_line(aes(y = dwpt, color = "Dew Point (C)"), linewidth = 0.25) +    # Dew point line third
  labs(title = "Hourly Weather Data 2018 & 2019",
       x = "Time", y = "Value") +
  scale_color_manual("",
                     breaks = c("Relative Humidity (RH)", "Temperature (C)", "Dew Point (C)"),  # Custom order
                     values = c("Relative Humidity (RH)" = "blue", 
                                "Temperature (C)" = "black", 
                                "Dew Point (C)" = "cyan")) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Electrical data in 2019 only (Target)
Energy_2019 <- read_csv("2019Floor2.csv", show_col_types = FALSE) %>%
               mutate(Date = as.POSIXct(Date))
colnames(Energy_2019)
kable(head(select(Energy_2019, 1:8)), format = "markdown", align = "l") 

# Function to sum minute electrical data (kW) to hourly energy data (kWh) 
sum_energy_kWh <- function(data) {
    data %>%
    mutate(Date = floor_date(Date, "hour")) %>%   # Round down to the hour
    group_by(Date) %>%                            # Group by rounded hour
    summarise(across(contains("kW"), sum, na.rm = TRUE)) %>%  # Sum all columns with "kW"
    rowwise() %>%
    mutate(total_Elec_kWh = sum(c_across(contains("kW")), na.rm = TRUE)) %>%
    ungroup() %>%
    select(Date, total_Elec_kWh)     # Select columns
}

# This part of code takes 3 minutes to run.
# Sum the electrical data 
Y2019_kWh <- sum_energy_kWh(Energy_2019)
kable(head(Y2019_kWh))

# Electrical Data Visualization 
ggplot(Y2019_kWh, aes(x = Date, y = total_Elec_kWh)) +
  geom_line(color = "blue") +
  labs(title = "Total Electricity Consumption 2019",
       x = "Date",
       y = "Total Electricity (kWh)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) 

# Insights: Monthly Pattern Plot
Y2019_kWh_month<- Y2019_kWh %>% mutate(Month = format(Date, "%m-%b"))

ggplot(Y2019_kWh_month, aes(x = Date, y = total_Elec_kWh)) +
  geom_line(color = "blue") +  
  labs(title = "Monthly Total Electricity Consumption in 2019",
       x = "Date",
       y = "Total Electricity (kWh)") +
  theme(axis.text.x = element_blank()) +  
  facet_wrap(~ Month, ncol = 4, scales = "free_x")

# Insights: Weekly Pattern Plot 
Y2019_kWh_week <- Y2019_kWh %>%
                  filter(Date >= as.Date("2019-01-14") & Date < as.Date("2019-01-20"))
ggplot(Y2019_kWh_week, aes(x = Date, y = total_Elec_kWh)) +
  geom_line(color = "blue") + 
  labs(title = "Total Electricity Consumption for a specific week in 2019",
       x = "Date",
       y = "Total Electricity (kWh)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#####################################################################################################
# Preparing data for training 
# Filter the weather data only for the training year 2019 
wea_2019 <- weather_2018_2019 %>% filter(format(time, "%Y") == "2019")

# Combine training and target data into single dataframe
combined_data <- left_join(Y2019_kWh, wea_2019, by = c("Date" = "time"))
kable(head(combined_data), format = "markdown", align = "l")

# Data split for train set and test set 
set.seed(123)
data_split <- initial_split(combined_data, prop = 0.8)
train_set <- training(data_split)
test_set <- testing(data_split)

#####################################################################################################
# Feature Engineering
# Function to process datetime (times of day - weekday - weekend - public holiday)
feature_engineering <- function(data, start_time, end_time, public_holidays) {
  data %>%
    mutate(Day_of_Week = as.numeric(factor(weekdays(Date), 
                                           levels = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"))),
           Is_Weekend = ifelse(Day_of_Week >= 6, 1, 0),  # Weekend indicator
           Is_Public_Holiday = ifelse(format(Date, "%m-%d") %in% public_holidays, 1, 0),  # Public holiday indicator
           Time_of_Day = hms::as_hms(format(Date, "%H:%M:%S")),  
           Is_Working_Time = ifelse(Is_Weekend == 0 &  # Check if it's a weekday
                                      Is_Public_Holiday == 0 &  # Check if it's not a public holiday
                                      Time_of_Day >= start_time & Time_of_Day <= end_time, 1, 0)) %>% # Check if time is within working hours
    mutate(Month = lubridate::month(Date),
           Day = lubridate::day(Date),
           Hour = lubridate::hour(Date)) %>%
    select(-Time_of_Day, -Month, -Day, -Hour)  
}

# Set Working hours
start_time <- parse_hms("08:30:00")
end_time <- parse_hms("16:30:00")

# Set National public holidays (month-day format only)
public_holidays <- c("01-01", 
                     "04-06", "04-13", "04-14", "04-15", 
                     "05-01", "05-04", 
                     "07-28",
                     "08-12",
                     "09-13", 
                     "12-05", "12-10", "12-31")

# Applying the feature engineering function to train and test set
train_fea <- feature_engineering(train_set, start_time, end_time, public_holidays)
test_fea <- feature_engineering(test_set, start_time, end_time, public_holidays)

#####################################################################################################
# LightGBM
###################################################################################################### 
# Function to train LightGBM and evaluate model 
train_Lightgbm <- function(train, test, 
                           target_col, params, nrounds = 100,  
                           early_stopping_rounds = 10) {
  train_matrix <- train %>%
    select(-Date, -!!sym(target_col)) %>%  # Exclude the target variable and Date column
    as.matrix()
  train_label <- train[[target_col]]  # Extract target variable
  
  test_matrix <- test %>%
    select(-Date, -!!sym(target_col)) %>%  # Exclude the target and Date column
    as.matrix()
  test_label <- test[[target_col]]  # Extract target column
  
  # Create LightGBM datasets
  dtrain_LightGBM <- lgb.Dataset(data = train_matrix, label = train_label)
  dtest_LightGBM <- lgb.Dataset(data = test_matrix, label = test_label, reference = dtrain_LightGBM)
  
  # Train the model    
  model_LightGBM <- lgb.train(
    params = params,
    data = dtrain_LightGBM,
    nrounds = nrounds,               
    valids = list(test = dtest_LightGBM),  
    early_stopping_rounds = early_stopping_rounds,
    verbose = -1)  
  
  # Predict on the test set
  preds <- predict(model_LightGBM, test_matrix)
  
  # Calculate RMSE and R-squared
  rmse <- sqrt(mean((preds- test_label)^2))
  ss_total <- sum((test_label - mean(test_label))^2)
  ss_residual <- sum((test_label - preds)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Return results
  return(list(
    model = model_LightGBM,
    rmse = rmse,
    r_squared = r_squared,
    predictions = preds))
}

############################
# Function to tune LightGBM
tune_Lightgbm <- function(train, test, target_col, tune_grid, nrounds = 100, early_stopping_rounds = 10) {
  results <- list()
  best_rmse <- Inf  
  best_model <- NULL 
  
  for (i in seq_len(nrow(tune_grid))) { # Loop through each combination in the tuning grid
    current_params <- list(
      objective = "regression",
      metric = "rmse",
      learning_rate = tune_grid$learning_rate[i],
      num_leaves = tune_grid$num_leaves[i],
      max_depth = tune_grid$max_depth[i])
    
    # Train the model 
    model_results <- train_Lightgbm(
      train = train,
      test = test,
      target_col = target_col,
      params = current_params,
      nrounds = nrounds,
      early_stopping_rounds = early_stopping_rounds)
    
    # Store the result 
    results[[i]] <- list(
      learning_rate = tune_grid$learning_rate[i],
      num_leaves = tune_grid$num_leaves[i],
      max_depth = tune_grid$max_depth[i],
      rmse = model_results$rmse,
      r_squared = model_results$r_squared)
    
    # Store the best model
    if (model_results$rmse < best_rmse) {
      best_rmse <- model_results$rmse
      best_model <- model_results$model  
    }
  }
  
  # Convert results list to dataframe 
  results_df <- do.call(rbind, lapply(results, as.data.frame))
  return(list(
    tuning_results = results_df,
    best_model = best_model)) 
}

#######################
# Tuning grid LightGBM
tune_grid <- expand.grid(learning_rate = c(0.01, 0.05, 0.1),  
                         num_leaves = c(20, 30, 40),          
                         max_depth = c(5, 10, 15))

# Call the tuning function
set.seed(456)
tuning_results_Lightgbm <- tune_Lightgbm(train_fea, test_fea, target_col = "total_Elec_kWh", tune_grid = tune_grid)

#print(tuning_results_Lightgbm)

# Find the best model and its params
best_rmse_LightGBM <- round(tuning_results_Lightgbm$tuning_result$rmse
                            [which.min(tuning_results_Lightgbm$tuning_results$rmse)],3)
best_r_squared_LightGBM <- round(tuning_results_Lightgbm$tuning_results$r_squared
                                 [which.max(tuning_results_Lightgbm$tuning_results$r_squared)], 3)

cat(paste("Best RMSE on test set (LightGBM):", best_rmse_LightGBM))
cat(paste("R-squared on test set (LightGBM):", best_r_squared_LightGBM))
# RMSE = 1087.875, R2 = 0.683

#####################################################################################################
# Random Forest
#####################################################################################################
# Function for checking missing values 
check_missing <- function(data, dataset_name = "Dataset") {
  if (anyNA(data)) {
    cat("Missing values found in", dataset_name, ":\n")
    print(colSums(is.na(data)))} 
  else {
    cat("No missing values in", dataset_name)}
}

# check_missing(train_fea, "train_fea")
# check_missing(test_fea, "test_fea")

train_fea_clean <- na.omit(train_fea)
test_fea_clean <- na.omit(test_fea)

check_missing(train_fea_clean, "train_fea_clean")
check_missing(test_fea_clean, "test_fea_clean")

#####################################################################################################
# Function to train Random Forest and evaluate model 
train_rf <- function(train, test, target_col, ntree = 500, mtry = 3) {
  train_matrix <- train %>%
    select(-Date, -!!sym(target_col))  # Exclude the target and Date column
  train_label <- train[[target_col]]  # Extract target column
  
  test_matrix <- test %>%
    select(-Date, -!!sym(target_col))  # Exclude the target and Date column
  test_label <- test[[target_col]]  # Extract target column
  
  # Train the model
  model_rf <- randomForest(
    x = train_matrix, 
    y = train_label, 
    ntree = ntree, 
    mtry = mtry)
  
  # Predict on the test set
  preds <- predict(model_rf, test_matrix)
  
  # Calculate RMSE and R-squared
  rmse<- sqrt(mean((preds - test_label)^2))
  ss_total <- sum((test_label - mean(test_label))^2)
  ss_residual <- sum((test_label - preds)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  
  # Return results
  return(list(
    model = model_rf,
    rmse = rmse,
    r_squared = r_squared,
    prediction = preds))
}

#################################
# Function to tune Random Forest 
tune_rf <- function(train, test, target_col, tune_grid) {
  results <- list()   
  for (i in seq_len(nrow(tune_grid))) { # Loop through each combination in the tuning grid
    current_mtry <- tune_grid$mtry[i]
    current_ntree <- tune_grid$ntree[i]
    
    # Train the  model
    model_results <- train_rf(
      train = train,
      test = test,
      target_col = target_col,
      ntree = current_ntree,
      mtry = current_mtry)
    
    # Store the result 
    results[[i]] <- list(
      mtry = current_mtry,
      ntree = current_ntree,
      rmse = model_results$rmse,
      r_squared = model_results$r_squared)
  }
  
  # Convert results list to a dataframe 
  results_df <- do.call(rbind, lapply(results, as.data.frame))
  return(results_df)
}

###########################
# Tuning grid Random Forest
# The code takes 3 minutes to run 
tune_grid <- expand.grid(mtry = c(2, 3, 4, 5),  
                         ntree = c(100, 200, 500))  

# Call the tuning function
set.seed(456)
tuning_results_rf <- tune_rf(train_fea_clean, test_fea_clean, target_col = "total_Elec_kWh", tune_grid = tune_grid)

# print(tuning_results_rf)

best_rmse_rf <- round(tuning_results_rf$rmse
                            [which.min(tuning_results_rf$rmse)],3)
best_r_squared_rf <- round(tuning_results_rf$r_squared
                                 [which.max(tuning_results_rf$r_squared)], 3)
cat(paste("Best RMSE on test set (Random Forest):", best_rmse_rf))
cat(paste("R-squared on test set (Random Forest):", best_r_squared_rf))
# RMSE = 1086.279, R2 = 0.684

#####################################################################################################
# Compare LightGBM and Random Forest
comparison <- data.frame(
  Model = c("LightGBM", "Random Forest"),
  RMSE = c(best_rmse_LightGBM, best_rmse_rf),
  R_squared = c(best_r_squared_LightGBM, best_r_squared_rf))
kable(comparison, caption = "Comparison of RMSE and R-squared")

#####################################################################################################
# Understand the best tuned LightGBM 
best_Lightgbm_model <- tuning_results_Lightgbm$best_model
print(best_Lightgbm_model)

# Extract feature importance from the best model
importance <- lgb.importance(best_Lightgbm_model, percentage = TRUE)
kable(print(importance)) 

# Extract params from the best model
best_params <- list(
  learning_rate = tuning_results_Lightgbm$best_model$params$learning_rate,
  num_leaves = tuning_results_Lightgbm$best_model$params$num_leaves,
  max_depth = tuning_results_Lightgbm$best_model$params$max_depth)

best_params_df <- as.data.frame(t(best_params))
kable(best_params_df, caption = "Best LightGBM Parameters")
# Best model LightGBM is learning_rate = 0.1, num_leaves = 20, max_depth = 5

######################################################################################################
# Adjust importance feature and retrain the model
start_time_imp <- parse_hms("08:00:00")
end_time_imp <- parse_hms("16:00:00")

train_fea_imp <- feature_engineering(train_set, start_time_imp, end_time_imp, public_holidays)
test_fea_imp <- feature_engineering(test_set, start_time_imp, end_time_imp, public_holidays)

set.seed(456)
result_imp <- train_Lightgbm(train_fea_imp, test_fea_imp, "total_Elec_kWh", best_params)

rmse_LightGBM_imp <- round(result_imp$rmse, 3)
r_squared_LightGBM_imp <- round(result_imp$r_squared, 3)

# Compare Best model and Model improvement 
comparison <- data.frame(
  Model = c("Best Tuned Params", "Feature Improvement"),
  RMSE = c(best_rmse_LightGBM, rmse_LightGBM_imp),
  R_squared = c(best_r_squared_LightGBM, r_squared_LightGBM_imp))
kable(comparison, caption = "Comparison of RMSE and R-squared")

#####################################################################################################
# Final evaluation on holdout test data 
# Read csv 
Energy_2018 <- read_csv("2018Floor2.csv", show_col_types = FALSE) %>%
  mutate(Date = as.POSIXct(Date))

# Preparing data follows training data 
Y2018_kWh <- sum_energy_kWh(Energy_2018)
wea_2018 <- weather_2018_2019 %>% filter(format(time, "%Y") == "2018")
combined_holdout <- left_join(Y2018_kWh, wea_2018, by = c("Date" = "time"))
holdout_fea <- feature_engineering(combined_holdout, start_time_imp, end_time_imp, public_holidays)

# Create matrix for LightGBM predictions
holdout_matrix <- holdout_fea %>%
  select(-Date, -!!sym("total_Elec_kWh")) %>%  # Exclude the target variable and Date column
  as.matrix()
holdout_label <- holdout_fea[["total_Elec_kWh"]] # Extract target variable

holdout_preds <- predict(best_Lightgbm_model,holdout_matrix)

# RMSE and R2 on holdout test data
holdout_rmse <- sqrt(mean((holdout_preds - holdout_label)^2))
holdout_ss_total <- sum((holdout_label - mean(holdout_label))^2)
holdout_ss_residual <- sum((holdout_label - holdout_preds)^2)
holdout_r_squared <- 1 - (holdout_ss_residual / holdout_ss_total)

cat("RMSE on holdout test set:", round(holdout_rmse,3))
cat("R-squared on holdout test set:", round(holdout_r_squared,3))

######################################################################################################
###################################### The End #######################################################
 
      