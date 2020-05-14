setwd("~/InterpretableHouseSalePrices/")

# load all (or more)
source("load_libs_data_explainers.R")

set.seed(1404)
s <- sample(nrow(train), 200)
sub_df <- train[s,]
sub_df_onehot <- train_onehot[s,]

# LATITUDE & LONGITUDE
cp_rf <- ceteris_paribus(explain_rf, sub_df)
cp_xgb <- ceteris_paribus(explain_xgb, sub_df)
cp_gbm1h <- ceteris_paribus(explain_gbm1h, sub_df_onehot)
cp_xgb1h <- ceteris_paribus(explain_xgb1h, sub_df_onehot)

pdp_rf <- aggregate_profiles(cp_rf)
pdp_xgb <- aggregate_profiles(cp_xgb)
pdp_gbm1h <- aggregate_profiles(cp_gbm1h)
pdp_xgb1h <- aggregate_profiles(cp_xgb1h)

save(pdp_rf, file ="pdp_rf.rda")
save(pdp_xgb, file ="pdp_xgb.rda")
save(pdp_xgb1h, file ="pdp_xgb1h.rda")
save(pdp_gbm1h, file ="pdp_gbm1h.rda")

plot(pdp_rf,pdp_xgb,pdp_gbm1h,pdp_xgb1h, variables = c("long","lat"))

plot(pdp_rf,pdp_xgb,pdp_gbm1h,pdp_xgb1h, variables = c("m2_living"))+xlim(50,500)
plot(pdp_rf,pdp_xgb,pdp_gbm1h,pdp_xgb1h, variables = c("m2_living","grade"))
+ xlim(0,20,50,500)

plot(pdp_rf,pdp_xgb, variables = "dist_stop") + xlim(0,2000)
plot(pdp_xgb,variables = "dist_stop")+ xlim(0,2000)
plot(pdp_gbm1h,variables = "dist_stop")+ xlim(0,2000)
plot(pdp_xgb1h,variables = "dist_stop")+ xlim(0,200)



ald_rf <- accumulated_dependence(explain_rf)
ald_xgb <- accumulated_dependence(explain_xgb)
ald_xgb1h <- accumulated_dependence(explain_xgb1h)
ald_gbm1h <- accumulated_dependence(explain_gbm1h)

save(ald_rf, file="./pd6/ald_rf.rda")
save(ald_xgb, file="./pd6/ald_xgb.rda")
save(ald_xgb1h, file="./pd6/ald_xgb1h.rda")
save(ald_gbm1h, file="./pd6/ald_gbm1h.rda")

plot(ald_rf)
plot(ald_xgb)
plot(ald_rf,ald_xgb, variables="dist_stop")+xlim(0,1000)
plot(ald_rf,ald_xgb, variables=c("long","lat"))
plot(ald_rf,ald_xgb, variables = "grade")
plot(ald_rf,ald_xgb, variables = "m2_living")+xlim(0,500)
plot(pdp_rf,pdp_xgb, variables = "m2_living")+xlim(0,500)


plot(
   pdp_gbm1h,
   pdp_xgb1h,
   variables = c(
      "zipcode98106",
      "zipcode98010",
      "zipcode98108",
      'zipcode98005'
   )
)

plot(pdp_rf,pdp_xgb,variables="zipcode")+xlim(98102,98108)

z98010 <- train[train$zipcode==98010,]
nrow(z98010)
z98106 <- train[train$zipcode==98106,]
nrow(z98106)
z98108 <- train[train$zipcode==98108,]
nrow(z98108)



tab <- train[train$zipcode %in% c(98010,98106,98108),c("zipcode","price_log","m2_living")]
par(mfrow=c(1,2))
boxplot(exp(price_log)~zipcode,data=tab,ylab = "Price",las=1)
boxplot(m2_living~zipcode,data=tab,ylab="m² (living)",las=1)


xt <- (z98010[z98010$price_log>13,])
xt$price = exp(xt$price_log)

row.names(xt)= 1:nrow(xt)
xt[5,]
View(xt)
