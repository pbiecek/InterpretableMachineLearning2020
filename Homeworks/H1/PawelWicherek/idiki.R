data <- read.csv("kc_house_data.csv")
idiki <- table(data$id)
print(length(idiki[idiki==2]))  # 175

#View(data[data$id == idiki[idiki==3],])
idiki_dup <- unique(as.numeric(names(idiki[idiki>1])))
duplicates <- (data[data$id %in% idiki_dup,])
#View(duplicates)


bez_dup <- data[!data$id %in% idiki_dup,]
write.csv(bez_dup,"kc_house_data_unique.csv")