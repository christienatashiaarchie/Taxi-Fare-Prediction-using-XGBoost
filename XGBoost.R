data <- read.csv("C:/Users/Christie Natashia/Desktop/AML-Code/Taxidata.csv")
data = data[1:1000,]

library(dplyr)
library(lubridate)
# Data Transformation
data = data %>%
  mutate (tpep_pickup_datetime = as.character(tpep_pickup_datetime),
          tpep_dropoff_datetime = as.character(tpep_dropoff_datetime),
          tpep_pickup_datetime = dmy_hm(tpep_pickup_datetime),
          tpep_dropoff_datetime = dmy_hm(tpep_dropoff_datetime),
          VendorID = factor(VendorID),
          passenger_count = factor(passenger_count),
          passenger_count = as.numeric(passenger_count),
          
  )



# Handling Abnormal Data
data = data %>%
  mutate(fare_amount = ifelse(fare_amount < 1,ave(fare_amount),fare_amount),
         extra = ifelse(extra < 1,ave(extra),extra),
         mta_tax = ifelse(mta_tax < 1,ave(mta_tax),mta_tax),
         tip_amount = ifelse(tip_amount < 1,ave(tip_amount),tip_amount),
         tolls_amount = ifelse(tolls_amount < 1,ave(tolls_amount),tolls_amount),
         total_amount = ifelse(total_amount  < 1,ave(total_amount ),total_amount),
         congestion_surcharge = ifelse(congestion_surcharge < 1,ave(congestion_surcharge),congestion_surcharge),
         tip_amount = ifelse(tip_amount < 1,ave(tip_amount),tip_amount),
         passenger_count = ifelse(passenger_count < 1,mode(passenger_count),passenger_count),
         improvement_surcharge = ifelse(improvement_surcharge < 1,ave(improvement_surcharge),improvement_surcharge),
         trip_distance = ifelse(trip_distance < 1,ave(trip_distance),trip_distance),
         store_and_fwd_flag = ifelse(store_and_fwd_flag == "Y", 1, 0)
  )
summary(data)




# Feature Engineering on Airport pickup and dropoff location
taxi_zone_lookup <- read.csv("https://s3.amazonaws.com/nyc-tlc/misc/taxi+_zone_lookup.csv")

airport_zone_id = c(1,132, 138) #Newark, JFK Airport, LaGuardia Airport
data = data %>%
  mutate(
    airport_pickup = ifelse(PULocationID %in% airport_zone_id, 1, 0),
    airport_dropoff = ifelse(DOLocationID %in% airport_zone_id, 1, 0)
  )

# Feature Engineering on date
library(chron)
library(lubridate)

data = data %>% 
  mutate(
    pickup_hour = hour(tpep_pickup_datetime),
    weekend = is.weekend(tpep_pickup_datetime),
    weekend = ifelse(weekend == TRUE, 1, 0)
  )


# Data Spliting
# install.packages("caTools")
library(caTools)
set.seed(123)

# Spliting
split = sample.split(data$total_amount, SplitRatio = 0.8)

train = subset(data, split == TRUE) # Train data
str(train)

test = subset(data, split == FALSE) # Test data 
str(test)


# XGBOOST
# install.packages('xgboost')
# install.packages('MLmetrics')
library(xgboost)
library(MLmetrics)

x_train = train %>% select(-total_amount, -tpep_pickup_datetime, -tpep_dropoff_datetime)
x_test = test %>% select(-total_amount, -tpep_pickup_datetime, -tpep_dropoff_datetime)
y_train = train$total_amount
y_test = test$total_amount


x_train = apply(x_train, 2, as.numeric)
x_test = apply(x_test, 2, as.numeric)


dtrain <- xgb.DMatrix(data=as.matrix(x_train), label=y_train)
dtest <- xgb.DMatrix(data=as.matrix(x_test), label=y_test)

# model 1
model_xgb1 <- xgb.cv(data=dtrain, nfold=10, nrounds=15,
                     objective = "reg:linear", verbose = 0)

model_xgb1$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean),
  )

# plot error vs number trees
ggplot(model_xgb1$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")

# ntrees.train rmse.train ntrees.test rmse.test
# 1           10   1.020139          10  1.484196

# ntrees.train rmse.train ntrees.test rmse.test
# 1           15  0.4103541          15  1.065951

# ntrees.train rmse.train ntrees.test rmse.test
# 1           20  0.2865822          20  1.080598

# ntrees.train rmse.train ntrees.test rmse.test
# 1           50  0.1086253          50  1.024723

# ntrees.train rmse.train ntrees.test rmse.test
# 1          100  0.0331833         100 0.9382971

# model 2
params = list(
  eta = .1,
  max_depth = 5,
  min_child_weight = 2,
  subsample = .8,
  colsample_bytree = .9,
  objective = "reg:linear"
)
  
set.seed(123)
 
model_xgb2 = xgb.cv( params = params, data = dtrain, nrounds =15, nfold =10,
                     verbose = 0, early_stopping_rounds = 10)

model_xgb2$evaluation_log %>%
  dplyr::summarise(
    ntrees.train = which(train_rmse_mean == min(train_rmse_mean))[1],
    rmse.train   = min(train_rmse_mean),
    ntrees.test  = which(test_rmse_mean == min(test_rmse_mean))[1],
    rmse.test   = min(test_rmse_mean),
  )

# plot error vs number trees
ggplot(model_xgb2$evaluation_log) +
  geom_line(aes(iter, train_rmse_mean), color = "red") +
  geom_line(aes(iter, test_rmse_mean), color = "blue")

# ntrees.train rmse.train ntrees.test rmse.test
# 1           15  0.5841904          15  1.081101


hyper_grid = expand.grid(
  eta = c(.1,.3),
  max_depth = c(1,3,5),
  min_child_weight = c(1,3,5),
  subsample = c( .8, 1),
  colsample_bytree = c(.8, 1),
  optimal_trees = 0,
  min_RMSE = 0
)

nrow(hyper_grid)

for(i in 1:nrow(hyper_grid)) {
  params = list (
    eta = hyper_grid$eta[i],
    max_depth = hyper_grid$max_depth[i],
    min_child_weight = hyper_grid$min_child_weight[i],
    subsample = hyper_grid$subsample[i],
    colsample_bytree = hyper_grid$colsample_bytree[i],
    nrounds = hyper_grid$nrounds[i],
    objective = "reg:linear"
  )
  set.seed(123)
  
  xgb.tune = xgb.cv(
    params = params, 
    data = dtrain, 
    nfold = 10,
    nrounds = 15,
    verbose = 0,
    early_stopping_rounds = 10
  )
  
  hyper_grid$optimal_trees[i] = which.min(xgb.tune$evaluation_log$test_rmse_mean)
  hyper_grid$min_RMSE[i] = min(xgb.tune$evaluation_log$test_rmse_mean)
}

hyper_grid %>%
  arrange(min_RMSE) %>%
  head(10)



# final model
final_params = list (
  eta = 0.3,
  max_depth = 5,
  min_child_weight = 3, 
  subsample = 0.8,
  colsample_bytree = 1.0,
  objective = "reg:linear"
)
final_xgb = xgboost(params = final_params, data = dtrain, nrounds = 20, verbose = 0,
                   early_stopping_rounds = 10)

min(final_xgb$evaluation_log$train_rmse)
# [1] 0.430988


y_pred = predict(final_xgb, dtest)

RMSE(y_pred, test$total_amount) 
# [1] 0.4939209
MAE(y_pred,test$total_amount)
# [1] 0.2847884

# to get Train MAE
x_pred = predict(final_xgb, dtrain)
MAE(y_pred,train$total_amount)
# [1] 0.2912938
