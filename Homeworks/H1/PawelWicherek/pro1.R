library("randomForest")
library(dplyr)
setwd("~/XAI/")

df <- read.csv("./kc_house_data_unique.csv")
View(head(df))
df$since_update = 2020 - pmax(df$yr_built,df$yr_renovated)
# df$zipcode = as.factor(df$zipcode)

dplyr::as_tibble(df)
df <- df %>% select(-c(id,date))

cor(df$zipcode,df$lat)
cor(df$zipcode,df$long)

y <- df$price
N <- length(y)
X <- df %>% select(-price)

set.seed(484)
s <- sample(N)
k <- round(0.7*N)
s_train <- s[1:k]
s_test <- s[-c(1:k)]

X_train <- X[s_train,]
y_train <- y[s_train]
X_test <- X[s_test,]
y_test <- y[s_test]

df_train <- df[s_train,]
df_test <- df[s_test,]

rf <- randomForest(price~.,df_train)
saveRDS(rf,file="rf_pred.rds")

tuned <- randomForest::tuneRF(X_train,y_train,doBest=TRUE)
tuned
saveRDS(tuned, file="rf_tuned.rds")
y_pred <- predict(tuned,newdata=df_test)
y_test <- df_test$price
mean(abs((y_pred- y_test)/y_test))

y_pred <- predict(rf,newdata=X_test)
y_test <- df_test$price
mean((y_pred - y_test)/y_test)


library(iBreakDown)
henry <- df[6,]
henry <- X_train[6,]
bd_rf <- break_down(rf, new_observation = henry, data=X_train)
plot(bd_rf)


