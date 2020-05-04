import pyreadr


train = pyreadr.read_r('hmc_train.Rda')['train']  # pandas df
valid = pyreadr.read_r('hmc_valid.Rda')['valid']

assert set(train.columns) == set(valid.columns)
column_names = sorted(train.columns)


column_names.remove("UNIQUE_ID")  # cannot influence

train = train[column_names]
valid = valid[column_names]

column_names.remove("PURCHASE")


def xy_split(data, y_name="PURCHASE"):
    return data.drop([y_name], axis=1).to_numpy(), data[y_name].to_numpy()


X_train, Y_train = xy_split(train)
X_valid, Y_valid = xy_split(valid)