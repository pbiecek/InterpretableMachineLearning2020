import pyreadr
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce

# feature eng
df = pd.read_csv('Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv')
df.drop(['history_segment', "conversion", "spend"], axis=1, inplace=True)
column_names = df.columns
cat_cols = ['zip_code', 'channel']
ce_one_hot = ce.OneHotEncoder(cols=cat_cols, use_cat_names=True)
data_ohe = ce_one_hot.fit_transform(df)
data_ohe.segment = data_ohe.segment.map({'Womens E-Mail': 1, 'Mens E-Mail': 1, 'No E-Mail': 0})
data = data_ohe.copy()
train = data_ohe.drop('visit', axis=1)
column_names = list(train.columns)
train_np = train.to_numpy().astype(float)
treatment_col = column_names.index('segment')
y = data_ohe.visit.to_numpy().astype(float)

X_train, X_valid, Y_train, Y_valid = train_test_split(train_np, y, test_size=0.2, stratify=y, random_state=42)

#train = pyreadr.read_r('hmc_train.Rda')['train']  # pandas df
#valid = pyreadr.read_r('hmc_valid.Rda')['valid']#
#assert set(train.columns) == set(valid.columns)
#column_names = sorted(train.columns)


#column_names.remove("UNIQUE_ID")  # cannot influence

#train = train[column_names]
#valid = valid[column_names]

#column_names.remove("PURCHASE")


#def xy_split(data, y_name="PURCHASE"):
#    return data.drop([y_name], axis=1).to_numpy(), data[y_name].to_numpy()


#X_train, Y_train = xy_split(train)
#X_valid, Y_valid = xy_split(valid)


