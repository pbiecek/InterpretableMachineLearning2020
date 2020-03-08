# Simple NN for survival prediction on the Titanic

author: Witalis Domitrz <witekdomitrz@gmail.com>

## Downloading the data


```
!wget http://students.mimuw.edu.pl/~wd393711/iml/titanic.zip
!unzip -o titanic.zip
!rm titanic.zip
```

    --2020-03-08 22:29:24--  http://students.mimuw.edu.pl/~wd393711/iml/titanic.zip
    Resolving students.mimuw.edu.pl (students.mimuw.edu.pl)... 193.0.96.129, 2001:6a0:5001:1::3
    Connecting to students.mimuw.edu.pl (students.mimuw.edu.pl)|193.0.96.129|:80... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 34877 (34K) [application/zip]
    Saving to: ‘titanic.zip’
    
    titanic.zip         100%[===================>]  34.06K   222KB/s    in 0.2s    
    
    2020-03-08 22:29:25 (222 KB/s) - ‘titanic.zip’ saved [34877/34877]
    
    Archive:  titanic.zip
      inflating: gender_submission.csv   
      inflating: test.csv                
      inflating: train.csv               


## Imports


```
%tensorflow_version 2.x
import numpy as np 
import pandas as pd
import tensorflow.keras.layers as layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import Callback
```

    TensorFlow 2.x selected.


## Prepare the data


```
X_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
Y_columns = ['Survived']

def load_data(fn):
    return pd.read_csv(fn).set_index('PassengerId')

def to_array(data):
    return pd.get_dummies(data).astype(dtype='float32').values

def split_to_x_y(data):
    return to_array(data[X_columns]), to_array(data[Y_columns])

def get_data():
    train = load_data('./train.csv')
    test = load_data('./gender_submission.csv').join(load_data('./test.csv'))    

    train['is_train'] = True
    test['is_train'] = False
    data = pd.concat([train, test])

    # Replace missing values with mean
    data.fillna(data.mean(), inplace=True)

    # Split test and train
    train = data[data['is_train']]
    test = data[data['is_train'] == False]

    return split_to_x_y(train), split_to_x_y(test)
```


```
(train_x, train_y), (test_x, test_y) = get_data()
```

## Create and train a model


```
def create_net():
    model = Sequential()
    for i in range(4):
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = Adadelta(lr=1.), metrics = ['accuracy'])
    return model
```


```
model = create_net()
batch_size = 256
training_history = model.fit(train_x, train_y, epochs=512, batch_size=batch_size, validation_split=0.2)
```

    Train on 712 samples, validate on 179 samples
    Epoch 1/512
    712/712 [==============================] - 2s 3ms/sample - loss: 0.8441 - accuracy: 0.5506 - val_loss: 0.8084 - val_accuracy: 0.3575
    Epoch 2/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.8920 - accuracy: 0.5028 - val_loss: 0.7446 - val_accuracy: 0.3128
    Epoch 3/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.8718 - accuracy: 0.5183 - val_loss: 0.7121 - val_accuracy: 0.3743
    Epoch 4/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.8304 - accuracy: 0.5520 - val_loss: 0.6912 - val_accuracy: 0.4581
    Epoch 5/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.8266 - accuracy: 0.5183 - val_loss: 0.6752 - val_accuracy: 0.6480
    Epoch 6/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.8003 - accuracy: 0.5421 - val_loss: 0.6686 - val_accuracy: 0.7095
    Epoch 7/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.8037 - accuracy: 0.5421 - val_loss: 0.6572 - val_accuracy: 0.6983
    Epoch 8/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.7721 - accuracy: 0.5548 - val_loss: 0.6505 - val_accuracy: 0.6927
    Epoch 9/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.8119 - accuracy: 0.5590 - val_loss: 0.6463 - val_accuracy: 0.6983
    Epoch 10/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7895 - accuracy: 0.5506 - val_loss: 0.6458 - val_accuracy: 0.6872
    Epoch 11/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.8079 - accuracy: 0.5407 - val_loss: 0.6437 - val_accuracy: 0.6816
    Epoch 12/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.8264 - accuracy: 0.5449 - val_loss: 0.6425 - val_accuracy: 0.6760
    Epoch 13/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.7625 - accuracy: 0.5702 - val_loss: 0.6375 - val_accuracy: 0.6816
    Epoch 14/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.7786 - accuracy: 0.5646 - val_loss: 0.6357 - val_accuracy: 0.6816
    Epoch 15/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7657 - accuracy: 0.5632 - val_loss: 0.6324 - val_accuracy: 0.6816
    Epoch 16/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.7658 - accuracy: 0.5702 - val_loss: 0.6308 - val_accuracy: 0.6816
    Epoch 17/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7580 - accuracy: 0.6011 - val_loss: 0.6315 - val_accuracy: 0.6816
    Epoch 18/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.7933 - accuracy: 0.5590 - val_loss: 0.6350 - val_accuracy: 0.6760
    Epoch 19/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.7451 - accuracy: 0.5885 - val_loss: 0.6363 - val_accuracy: 0.6760
    Epoch 20/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7192 - accuracy: 0.5955 - val_loss: 0.6344 - val_accuracy: 0.6760
    Epoch 21/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7118 - accuracy: 0.6208 - val_loss: 0.6365 - val_accuracy: 0.6704
    Epoch 22/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7326 - accuracy: 0.5730 - val_loss: 0.6377 - val_accuracy: 0.6704
    Epoch 23/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.7341 - accuracy: 0.5885 - val_loss: 0.6344 - val_accuracy: 0.6760
    Epoch 24/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7524 - accuracy: 0.5618 - val_loss: 0.6350 - val_accuracy: 0.6760
    Epoch 25/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7533 - accuracy: 0.5997 - val_loss: 0.6391 - val_accuracy: 0.6704
    Epoch 26/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7559 - accuracy: 0.5871 - val_loss: 0.6409 - val_accuracy: 0.6704
    Epoch 27/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7525 - accuracy: 0.6039 - val_loss: 0.6359 - val_accuracy: 0.6704
    Epoch 28/512
    712/712 [==============================] - 0s 49us/sample - loss: 0.6979 - accuracy: 0.6138 - val_loss: 0.6347 - val_accuracy: 0.6760
    Epoch 29/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.7120 - accuracy: 0.6011 - val_loss: 0.6323 - val_accuracy: 0.6760
    Epoch 30/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.7248 - accuracy: 0.5660 - val_loss: 0.6343 - val_accuracy: 0.6760
    Epoch 31/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.7000 - accuracy: 0.6067 - val_loss: 0.6324 - val_accuracy: 0.6760
    Epoch 32/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.7199 - accuracy: 0.5997 - val_loss: 0.6344 - val_accuracy: 0.6760
    Epoch 33/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.7347 - accuracy: 0.5983 - val_loss: 0.6351 - val_accuracy: 0.6760
    Epoch 34/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.7227 - accuracy: 0.5843 - val_loss: 0.6336 - val_accuracy: 0.6760
    Epoch 35/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.7035 - accuracy: 0.6278 - val_loss: 0.6309 - val_accuracy: 0.6760
    Epoch 36/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7161 - accuracy: 0.6025 - val_loss: 0.6262 - val_accuracy: 0.6760
    Epoch 37/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7085 - accuracy: 0.6067 - val_loss: 0.6275 - val_accuracy: 0.6760
    Epoch 38/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6914 - accuracy: 0.6320 - val_loss: 0.6267 - val_accuracy: 0.6760
    Epoch 39/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7199 - accuracy: 0.5730 - val_loss: 0.6277 - val_accuracy: 0.6760
    Epoch 40/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.7113 - accuracy: 0.6081 - val_loss: 0.6249 - val_accuracy: 0.6760
    Epoch 41/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.7205 - accuracy: 0.5871 - val_loss: 0.6229 - val_accuracy: 0.6760
    Epoch 42/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6977 - accuracy: 0.6152 - val_loss: 0.6232 - val_accuracy: 0.6760
    Epoch 43/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6922 - accuracy: 0.6180 - val_loss: 0.6219 - val_accuracy: 0.6760
    Epoch 44/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7414 - accuracy: 0.5801 - val_loss: 0.6232 - val_accuracy: 0.6760
    Epoch 45/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7071 - accuracy: 0.6011 - val_loss: 0.6238 - val_accuracy: 0.6760
    Epoch 46/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7339 - accuracy: 0.5772 - val_loss: 0.6253 - val_accuracy: 0.6760
    Epoch 47/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7056 - accuracy: 0.6053 - val_loss: 0.6282 - val_accuracy: 0.6760
    Epoch 48/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.7100 - accuracy: 0.6096 - val_loss: 0.6314 - val_accuracy: 0.6760
    Epoch 49/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7088 - accuracy: 0.6081 - val_loss: 0.6326 - val_accuracy: 0.6760
    Epoch 50/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6769 - accuracy: 0.6320 - val_loss: 0.6310 - val_accuracy: 0.6760
    Epoch 51/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.7163 - accuracy: 0.6025 - val_loss: 0.6291 - val_accuracy: 0.6760
    Epoch 52/512
    712/712 [==============================] - 0s 32us/sample - loss: 0.7125 - accuracy: 0.6124 - val_loss: 0.6296 - val_accuracy: 0.6760
    Epoch 53/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6725 - accuracy: 0.6208 - val_loss: 0.6283 - val_accuracy: 0.6760
    Epoch 54/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6899 - accuracy: 0.6124 - val_loss: 0.6264 - val_accuracy: 0.6760
    Epoch 55/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.7035 - accuracy: 0.6081 - val_loss: 0.6265 - val_accuracy: 0.6760
    Epoch 56/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.7202 - accuracy: 0.6138 - val_loss: 0.6298 - val_accuracy: 0.6760
    Epoch 57/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6973 - accuracy: 0.6067 - val_loss: 0.6304 - val_accuracy: 0.6760
    Epoch 58/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6955 - accuracy: 0.6180 - val_loss: 0.6301 - val_accuracy: 0.6760
    Epoch 59/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.7169 - accuracy: 0.5927 - val_loss: 0.6297 - val_accuracy: 0.6760
    Epoch 60/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.7280 - accuracy: 0.5857 - val_loss: 0.6293 - val_accuracy: 0.6760
    Epoch 61/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6964 - accuracy: 0.6222 - val_loss: 0.6290 - val_accuracy: 0.6760
    Epoch 62/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.7265 - accuracy: 0.5885 - val_loss: 0.6280 - val_accuracy: 0.6760
    Epoch 63/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.6883 - accuracy: 0.6025 - val_loss: 0.6291 - val_accuracy: 0.6760
    Epoch 64/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6957 - accuracy: 0.6053 - val_loss: 0.6278 - val_accuracy: 0.6760
    Epoch 65/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.7082 - accuracy: 0.6011 - val_loss: 0.6250 - val_accuracy: 0.6760
    Epoch 66/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.7061 - accuracy: 0.6025 - val_loss: 0.6220 - val_accuracy: 0.6760
    Epoch 67/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6983 - accuracy: 0.6166 - val_loss: 0.6207 - val_accuracy: 0.6816
    Epoch 68/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6883 - accuracy: 0.6138 - val_loss: 0.6215 - val_accuracy: 0.6816
    Epoch 69/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6722 - accuracy: 0.6166 - val_loss: 0.6223 - val_accuracy: 0.6816
    Epoch 70/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6940 - accuracy: 0.6180 - val_loss: 0.6229 - val_accuracy: 0.6816
    Epoch 71/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6982 - accuracy: 0.6110 - val_loss: 0.6248 - val_accuracy: 0.6760
    Epoch 72/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6739 - accuracy: 0.6334 - val_loss: 0.6244 - val_accuracy: 0.6760
    Epoch 73/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6824 - accuracy: 0.6222 - val_loss: 0.6236 - val_accuracy: 0.6816
    Epoch 74/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6891 - accuracy: 0.6081 - val_loss: 0.6236 - val_accuracy: 0.6760
    Epoch 75/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.6550 - accuracy: 0.6433 - val_loss: 0.6220 - val_accuracy: 0.6816
    Epoch 76/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6589 - accuracy: 0.6433 - val_loss: 0.6213 - val_accuracy: 0.6816
    Epoch 77/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6713 - accuracy: 0.6419 - val_loss: 0.6206 - val_accuracy: 0.6760
    Epoch 78/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.7011 - accuracy: 0.6096 - val_loss: 0.6213 - val_accuracy: 0.6816
    Epoch 79/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6821 - accuracy: 0.6236 - val_loss: 0.6196 - val_accuracy: 0.6816
    Epoch 80/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6605 - accuracy: 0.6306 - val_loss: 0.6194 - val_accuracy: 0.6816
    Epoch 81/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6901 - accuracy: 0.6250 - val_loss: 0.6207 - val_accuracy: 0.6816
    Epoch 82/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6655 - accuracy: 0.6236 - val_loss: 0.6208 - val_accuracy: 0.6816
    Epoch 83/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6897 - accuracy: 0.6039 - val_loss: 0.6184 - val_accuracy: 0.6872
    Epoch 84/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6940 - accuracy: 0.6376 - val_loss: 0.6188 - val_accuracy: 0.6872
    Epoch 85/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6860 - accuracy: 0.6110 - val_loss: 0.6183 - val_accuracy: 0.6872
    Epoch 86/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6860 - accuracy: 0.6180 - val_loss: 0.6181 - val_accuracy: 0.6872
    Epoch 87/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6654 - accuracy: 0.6447 - val_loss: 0.6184 - val_accuracy: 0.6872
    Epoch 88/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6906 - accuracy: 0.6334 - val_loss: 0.6176 - val_accuracy: 0.6872
    Epoch 89/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6720 - accuracy: 0.6208 - val_loss: 0.6174 - val_accuracy: 0.6872
    Epoch 90/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6838 - accuracy: 0.6138 - val_loss: 0.6168 - val_accuracy: 0.6872
    Epoch 91/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6711 - accuracy: 0.6292 - val_loss: 0.6152 - val_accuracy: 0.6872
    Epoch 92/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6693 - accuracy: 0.6348 - val_loss: 0.6149 - val_accuracy: 0.6872
    Epoch 93/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6972 - accuracy: 0.6180 - val_loss: 0.6169 - val_accuracy: 0.6872
    Epoch 94/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6754 - accuracy: 0.6264 - val_loss: 0.6192 - val_accuracy: 0.6872
    Epoch 95/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.6739 - accuracy: 0.6180 - val_loss: 0.6175 - val_accuracy: 0.6872
    Epoch 96/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.6528 - accuracy: 0.6461 - val_loss: 0.6158 - val_accuracy: 0.6872
    Epoch 97/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6620 - accuracy: 0.6264 - val_loss: 0.6164 - val_accuracy: 0.6872
    Epoch 98/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6741 - accuracy: 0.6124 - val_loss: 0.6149 - val_accuracy: 0.6872
    Epoch 99/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6704 - accuracy: 0.6390 - val_loss: 0.6139 - val_accuracy: 0.6872
    Epoch 100/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6666 - accuracy: 0.6292 - val_loss: 0.6135 - val_accuracy: 0.6872
    Epoch 101/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6727 - accuracy: 0.6208 - val_loss: 0.6131 - val_accuracy: 0.6872
    Epoch 102/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6723 - accuracy: 0.6250 - val_loss: 0.6127 - val_accuracy: 0.6872
    Epoch 103/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6612 - accuracy: 0.6320 - val_loss: 0.6124 - val_accuracy: 0.6927
    Epoch 104/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6794 - accuracy: 0.6180 - val_loss: 0.6148 - val_accuracy: 0.6872
    Epoch 105/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.6520 - accuracy: 0.6250 - val_loss: 0.6141 - val_accuracy: 0.6927
    Epoch 106/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6751 - accuracy: 0.6292 - val_loss: 0.6133 - val_accuracy: 0.6927
    Epoch 107/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.6706 - accuracy: 0.6390 - val_loss: 0.6131 - val_accuracy: 0.6872
    Epoch 108/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6876 - accuracy: 0.6236 - val_loss: 0.6123 - val_accuracy: 0.6927
    Epoch 109/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6785 - accuracy: 0.6222 - val_loss: 0.6118 - val_accuracy: 0.6927
    Epoch 110/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6688 - accuracy: 0.6222 - val_loss: 0.6128 - val_accuracy: 0.6872
    Epoch 111/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6617 - accuracy: 0.6362 - val_loss: 0.6141 - val_accuracy: 0.6872
    Epoch 112/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6470 - accuracy: 0.6503 - val_loss: 0.6116 - val_accuracy: 0.6927
    Epoch 113/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6639 - accuracy: 0.6419 - val_loss: 0.6113 - val_accuracy: 0.6983
    Epoch 114/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6647 - accuracy: 0.6376 - val_loss: 0.6093 - val_accuracy: 0.6983
    Epoch 115/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6515 - accuracy: 0.6306 - val_loss: 0.6074 - val_accuracy: 0.7039
    Epoch 116/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6618 - accuracy: 0.6306 - val_loss: 0.6077 - val_accuracy: 0.7039
    Epoch 117/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6631 - accuracy: 0.6433 - val_loss: 0.6082 - val_accuracy: 0.6983
    Epoch 118/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6522 - accuracy: 0.6334 - val_loss: 0.6072 - val_accuracy: 0.7039
    Epoch 119/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6521 - accuracy: 0.6419 - val_loss: 0.6057 - val_accuracy: 0.7095
    Epoch 120/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6613 - accuracy: 0.6152 - val_loss: 0.6043 - val_accuracy: 0.7095
    Epoch 121/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.6509 - accuracy: 0.6376 - val_loss: 0.6039 - val_accuracy: 0.7095
    Epoch 122/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.6687 - accuracy: 0.6348 - val_loss: 0.6010 - val_accuracy: 0.7039
    Epoch 123/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6546 - accuracy: 0.6376 - val_loss: 0.6012 - val_accuracy: 0.7039
    Epoch 124/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6483 - accuracy: 0.6531 - val_loss: 0.6014 - val_accuracy: 0.7039
    Epoch 125/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6405 - accuracy: 0.6475 - val_loss: 0.6012 - val_accuracy: 0.7039
    Epoch 126/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.6627 - accuracy: 0.6362 - val_loss: 0.5997 - val_accuracy: 0.7039
    Epoch 127/512
    712/712 [==============================] - 0s 31us/sample - loss: 0.6514 - accuracy: 0.6475 - val_loss: 0.5989 - val_accuracy: 0.7039
    Epoch 128/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6606 - accuracy: 0.6320 - val_loss: 0.5984 - val_accuracy: 0.7039
    Epoch 129/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6542 - accuracy: 0.6376 - val_loss: 0.5987 - val_accuracy: 0.7039
    Epoch 130/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6527 - accuracy: 0.6404 - val_loss: 0.5985 - val_accuracy: 0.7039
    Epoch 131/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6514 - accuracy: 0.6390 - val_loss: 0.5972 - val_accuracy: 0.7039
    Epoch 132/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6481 - accuracy: 0.6404 - val_loss: 0.5961 - val_accuracy: 0.7151
    Epoch 133/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6494 - accuracy: 0.6657 - val_loss: 0.5960 - val_accuracy: 0.7095
    Epoch 134/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.6572 - accuracy: 0.6348 - val_loss: 0.5980 - val_accuracy: 0.7039
    Epoch 135/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6731 - accuracy: 0.6461 - val_loss: 0.5966 - val_accuracy: 0.7039
    Epoch 136/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6563 - accuracy: 0.6376 - val_loss: 0.5981 - val_accuracy: 0.7039
    Epoch 137/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.6547 - accuracy: 0.6320 - val_loss: 0.5986 - val_accuracy: 0.7095
    Epoch 138/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6492 - accuracy: 0.6334 - val_loss: 0.5971 - val_accuracy: 0.7039
    Epoch 139/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6647 - accuracy: 0.6306 - val_loss: 0.5976 - val_accuracy: 0.7095
    Epoch 140/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6494 - accuracy: 0.6404 - val_loss: 0.5947 - val_accuracy: 0.7151
    Epoch 141/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6397 - accuracy: 0.6517 - val_loss: 0.5923 - val_accuracy: 0.7151
    Epoch 142/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6547 - accuracy: 0.6334 - val_loss: 0.5912 - val_accuracy: 0.7151
    Epoch 143/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6635 - accuracy: 0.6264 - val_loss: 0.5912 - val_accuracy: 0.7151
    Epoch 144/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6435 - accuracy: 0.6461 - val_loss: 0.5905 - val_accuracy: 0.7151
    Epoch 145/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6536 - accuracy: 0.6447 - val_loss: 0.5930 - val_accuracy: 0.7039
    Epoch 146/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6427 - accuracy: 0.6489 - val_loss: 0.5924 - val_accuracy: 0.7039
    Epoch 147/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6529 - accuracy: 0.6334 - val_loss: 0.5908 - val_accuracy: 0.7039
    Epoch 148/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6618 - accuracy: 0.6264 - val_loss: 0.5886 - val_accuracy: 0.7151
    Epoch 149/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6409 - accuracy: 0.6390 - val_loss: 0.5890 - val_accuracy: 0.7039
    Epoch 150/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6482 - accuracy: 0.6601 - val_loss: 0.5900 - val_accuracy: 0.7095
    Epoch 151/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6510 - accuracy: 0.6545 - val_loss: 0.5893 - val_accuracy: 0.7039
    Epoch 152/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6439 - accuracy: 0.6404 - val_loss: 0.5871 - val_accuracy: 0.7095
    Epoch 153/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6465 - accuracy: 0.6601 - val_loss: 0.5860 - val_accuracy: 0.7095
    Epoch 154/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6336 - accuracy: 0.6643 - val_loss: 0.5855 - val_accuracy: 0.7095
    Epoch 155/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6413 - accuracy: 0.6601 - val_loss: 0.5836 - val_accuracy: 0.7095
    Epoch 156/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.6327 - accuracy: 0.6559 - val_loss: 0.5816 - val_accuracy: 0.7095
    Epoch 157/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.6402 - accuracy: 0.6503 - val_loss: 0.5769 - val_accuracy: 0.7095
    Epoch 158/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6304 - accuracy: 0.6587 - val_loss: 0.5737 - val_accuracy: 0.7151
    Epoch 159/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.6341 - accuracy: 0.6601 - val_loss: 0.5811 - val_accuracy: 0.6983
    Epoch 160/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6256 - accuracy: 0.6699 - val_loss: 0.5686 - val_accuracy: 0.7207
    Epoch 161/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6171 - accuracy: 0.6671 - val_loss: 0.5749 - val_accuracy: 0.7151
    Epoch 162/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6222 - accuracy: 0.6601 - val_loss: 0.5643 - val_accuracy: 0.7207
    Epoch 163/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6221 - accuracy: 0.6573 - val_loss: 0.5745 - val_accuracy: 0.7151
    Epoch 164/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6299 - accuracy: 0.6629 - val_loss: 0.5832 - val_accuracy: 0.6927
    Epoch 165/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.6368 - accuracy: 0.6629 - val_loss: 0.5716 - val_accuracy: 0.7151
    Epoch 166/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6163 - accuracy: 0.6812 - val_loss: 0.5656 - val_accuracy: 0.7151
    Epoch 167/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6144 - accuracy: 0.6784 - val_loss: 0.5552 - val_accuracy: 0.7207
    Epoch 168/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.6103 - accuracy: 0.6671 - val_loss: 0.5594 - val_accuracy: 0.7207
    Epoch 169/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.6160 - accuracy: 0.6784 - val_loss: 0.5762 - val_accuracy: 0.6927
    Epoch 170/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6219 - accuracy: 0.6784 - val_loss: 0.5687 - val_accuracy: 0.7039
    Epoch 171/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.6065 - accuracy: 0.6713 - val_loss: 0.5481 - val_accuracy: 0.7263
    Epoch 172/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.6149 - accuracy: 0.6587 - val_loss: 0.5462 - val_accuracy: 0.7263
    Epoch 173/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.6007 - accuracy: 0.6868 - val_loss: 0.5380 - val_accuracy: 0.7374
    Epoch 174/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.5971 - accuracy: 0.6896 - val_loss: 0.5445 - val_accuracy: 0.7151
    Epoch 175/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.6142 - accuracy: 0.6756 - val_loss: 0.5363 - val_accuracy: 0.7374
    Epoch 176/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.6007 - accuracy: 0.6840 - val_loss: 0.5543 - val_accuracy: 0.7151
    Epoch 177/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5924 - accuracy: 0.7163 - val_loss: 0.5569 - val_accuracy: 0.6983
    Epoch 178/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5899 - accuracy: 0.6980 - val_loss: 0.5362 - val_accuracy: 0.7263
    Epoch 179/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.5706 - accuracy: 0.7191 - val_loss: 0.5300 - val_accuracy: 0.7374
    Epoch 180/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.5770 - accuracy: 0.7037 - val_loss: 0.5112 - val_accuracy: 0.7821
    Epoch 181/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5877 - accuracy: 0.7149 - val_loss: 0.5653 - val_accuracy: 0.6927
    Epoch 182/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.5523 - accuracy: 0.7317 - val_loss: 0.5355 - val_accuracy: 0.7486
    Epoch 183/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5748 - accuracy: 0.7219 - val_loss: 0.5176 - val_accuracy: 0.7654
    Epoch 184/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.5889 - accuracy: 0.7079 - val_loss: 0.5063 - val_accuracy: 0.7877
    Epoch 185/512
    712/712 [==============================] - 0s 49us/sample - loss: 0.5865 - accuracy: 0.6952 - val_loss: 0.4902 - val_accuracy: 0.7933
    Epoch 186/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.5687 - accuracy: 0.7191 - val_loss: 0.4892 - val_accuracy: 0.7821
    Epoch 187/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5786 - accuracy: 0.7149 - val_loss: 0.4840 - val_accuracy: 0.7877
    Epoch 188/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5463 - accuracy: 0.7317 - val_loss: 0.5165 - val_accuracy: 0.7877
    Epoch 189/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.5481 - accuracy: 0.7261 - val_loss: 0.4849 - val_accuracy: 0.7989
    Epoch 190/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.5519 - accuracy: 0.7331 - val_loss: 0.4808 - val_accuracy: 0.7933
    Epoch 191/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5843 - accuracy: 0.7163 - val_loss: 0.4705 - val_accuracy: 0.7821
    Epoch 192/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5644 - accuracy: 0.7374 - val_loss: 0.5569 - val_accuracy: 0.7430
    Epoch 193/512
    712/712 [==============================] - 0s 53us/sample - loss: 0.5498 - accuracy: 0.7388 - val_loss: 0.5121 - val_accuracy: 0.7877
    Epoch 194/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5441 - accuracy: 0.7388 - val_loss: 0.4747 - val_accuracy: 0.8101
    Epoch 195/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5499 - accuracy: 0.7331 - val_loss: 0.5053 - val_accuracy: 0.7821
    Epoch 196/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5410 - accuracy: 0.7360 - val_loss: 0.4678 - val_accuracy: 0.8212
    Epoch 197/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5567 - accuracy: 0.7402 - val_loss: 0.4622 - val_accuracy: 0.7989
    Epoch 198/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.5582 - accuracy: 0.7346 - val_loss: 0.4694 - val_accuracy: 0.8156
    Epoch 199/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.5457 - accuracy: 0.7528 - val_loss: 0.4489 - val_accuracy: 0.7933
    Epoch 200/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5529 - accuracy: 0.7261 - val_loss: 0.4510 - val_accuracy: 0.7933
    Epoch 201/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5223 - accuracy: 0.7402 - val_loss: 0.4484 - val_accuracy: 0.7877
    Epoch 202/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5296 - accuracy: 0.7444 - val_loss: 0.4512 - val_accuracy: 0.7933
    Epoch 203/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5448 - accuracy: 0.7430 - val_loss: 0.5768 - val_accuracy: 0.7374
    Epoch 204/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5629 - accuracy: 0.7416 - val_loss: 0.5376 - val_accuracy: 0.7486
    Epoch 205/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.5232 - accuracy: 0.7640 - val_loss: 0.4653 - val_accuracy: 0.8324
    Epoch 206/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5360 - accuracy: 0.7556 - val_loss: 0.4666 - val_accuracy: 0.8212
    Epoch 207/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.5321 - accuracy: 0.7626 - val_loss: 0.4533 - val_accuracy: 0.8268
    Epoch 208/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5270 - accuracy: 0.7500 - val_loss: 0.5068 - val_accuracy: 0.7877
    Epoch 209/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5285 - accuracy: 0.7486 - val_loss: 0.4713 - val_accuracy: 0.8268
    Epoch 210/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5309 - accuracy: 0.7556 - val_loss: 0.4439 - val_accuracy: 0.8101
    Epoch 211/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5315 - accuracy: 0.7640 - val_loss: 0.4499 - val_accuracy: 0.8101
    Epoch 212/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5347 - accuracy: 0.7486 - val_loss: 0.4782 - val_accuracy: 0.8268
    Epoch 213/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5246 - accuracy: 0.7556 - val_loss: 0.5529 - val_accuracy: 0.7709
    Epoch 214/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.5362 - accuracy: 0.7486 - val_loss: 0.4853 - val_accuracy: 0.7989
    Epoch 215/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.5150 - accuracy: 0.7823 - val_loss: 0.5627 - val_accuracy: 0.7374
    Epoch 216/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.5303 - accuracy: 0.7542 - val_loss: 0.5237 - val_accuracy: 0.7654
    Epoch 217/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5316 - accuracy: 0.7626 - val_loss: 0.5375 - val_accuracy: 0.7542
    Epoch 218/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5347 - accuracy: 0.7598 - val_loss: 0.4657 - val_accuracy: 0.8212
    Epoch 219/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5278 - accuracy: 0.7500 - val_loss: 0.4774 - val_accuracy: 0.8212
    Epoch 220/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5467 - accuracy: 0.7317 - val_loss: 0.5431 - val_accuracy: 0.7430
    Epoch 221/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5510 - accuracy: 0.7514 - val_loss: 0.6007 - val_accuracy: 0.7095
    Epoch 222/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5321 - accuracy: 0.7669 - val_loss: 0.4829 - val_accuracy: 0.8045
    Epoch 223/512
    712/712 [==============================] - 0s 47us/sample - loss: 0.5255 - accuracy: 0.7598 - val_loss: 0.4637 - val_accuracy: 0.8268
    Epoch 224/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5104 - accuracy: 0.7823 - val_loss: 0.5222 - val_accuracy: 0.7542
    Epoch 225/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5143 - accuracy: 0.7626 - val_loss: 0.5958 - val_accuracy: 0.7318
    Epoch 226/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5271 - accuracy: 0.7528 - val_loss: 0.5048 - val_accuracy: 0.7765
    Epoch 227/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.5230 - accuracy: 0.7640 - val_loss: 0.5480 - val_accuracy: 0.7765
    Epoch 228/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5098 - accuracy: 0.7640 - val_loss: 0.5546 - val_accuracy: 0.7542
    Epoch 229/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5130 - accuracy: 0.7683 - val_loss: 0.5228 - val_accuracy: 0.7654
    Epoch 230/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5338 - accuracy: 0.7528 - val_loss: 0.4868 - val_accuracy: 0.8101
    Epoch 231/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.5182 - accuracy: 0.7823 - val_loss: 0.4452 - val_accuracy: 0.8268
    Epoch 232/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4977 - accuracy: 0.7935 - val_loss: 0.4380 - val_accuracy: 0.8436
    Epoch 233/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5016 - accuracy: 0.7626 - val_loss: 0.4645 - val_accuracy: 0.8156
    Epoch 234/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5275 - accuracy: 0.7598 - val_loss: 0.4807 - val_accuracy: 0.8101
    Epoch 235/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.5093 - accuracy: 0.7725 - val_loss: 0.4794 - val_accuracy: 0.8268
    Epoch 236/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.5008 - accuracy: 0.7879 - val_loss: 0.4916 - val_accuracy: 0.7989
    Epoch 237/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4788 - accuracy: 0.7978 - val_loss: 0.4854 - val_accuracy: 0.8045
    Epoch 238/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5183 - accuracy: 0.7612 - val_loss: 0.5120 - val_accuracy: 0.7821
    Epoch 239/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.5174 - accuracy: 0.7556 - val_loss: 0.5419 - val_accuracy: 0.7542
    Epoch 240/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5197 - accuracy: 0.7683 - val_loss: 0.5231 - val_accuracy: 0.7765
    Epoch 241/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.5253 - accuracy: 0.7584 - val_loss: 0.4520 - val_accuracy: 0.8212
    Epoch 242/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5199 - accuracy: 0.7795 - val_loss: 0.5400 - val_accuracy: 0.7430
    Epoch 243/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.5220 - accuracy: 0.7739 - val_loss: 0.4743 - val_accuracy: 0.8156
    Epoch 244/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5124 - accuracy: 0.7753 - val_loss: 0.4576 - val_accuracy: 0.8268
    Epoch 245/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.5236 - accuracy: 0.7570 - val_loss: 0.5103 - val_accuracy: 0.7821
    Epoch 246/512
    712/712 [==============================] - 0s 32us/sample - loss: 0.4968 - accuracy: 0.7879 - val_loss: 0.5454 - val_accuracy: 0.7598
    Epoch 247/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.5110 - accuracy: 0.7725 - val_loss: 0.4304 - val_accuracy: 0.8156
    Epoch 248/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.5227 - accuracy: 0.7570 - val_loss: 0.4259 - val_accuracy: 0.8268
    Epoch 249/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.5159 - accuracy: 0.7669 - val_loss: 0.4307 - val_accuracy: 0.8268
    Epoch 250/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4810 - accuracy: 0.7823 - val_loss: 0.5239 - val_accuracy: 0.7486
    Epoch 251/512
    712/712 [==============================] - 0s 33us/sample - loss: 0.5028 - accuracy: 0.7907 - val_loss: 0.5156 - val_accuracy: 0.7654
    Epoch 252/512
    712/712 [==============================] - 0s 33us/sample - loss: 0.5080 - accuracy: 0.7683 - val_loss: 0.4648 - val_accuracy: 0.8156
    Epoch 253/512
    712/712 [==============================] - 0s 33us/sample - loss: 0.4836 - accuracy: 0.7739 - val_loss: 0.4682 - val_accuracy: 0.8156
    Epoch 254/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4846 - accuracy: 0.7753 - val_loss: 0.4521 - val_accuracy: 0.8045
    Epoch 255/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.5129 - accuracy: 0.7809 - val_loss: 0.4802 - val_accuracy: 0.7989
    Epoch 256/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4821 - accuracy: 0.7809 - val_loss: 0.4594 - val_accuracy: 0.8156
    Epoch 257/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4828 - accuracy: 0.7963 - val_loss: 0.5075 - val_accuracy: 0.7709
    Epoch 258/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4726 - accuracy: 0.8020 - val_loss: 0.4615 - val_accuracy: 0.8212
    Epoch 259/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4767 - accuracy: 0.7921 - val_loss: 0.4741 - val_accuracy: 0.7989
    Epoch 260/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4998 - accuracy: 0.7725 - val_loss: 0.4482 - val_accuracy: 0.8324
    Epoch 261/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4781 - accuracy: 0.7949 - val_loss: 0.4549 - val_accuracy: 0.8212
    Epoch 262/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4847 - accuracy: 0.7907 - val_loss: 0.4267 - val_accuracy: 0.8268
    Epoch 263/512
    712/712 [==============================] - 0s 50us/sample - loss: 0.5003 - accuracy: 0.7823 - val_loss: 0.4242 - val_accuracy: 0.8212
    Epoch 264/512
    712/712 [==============================] - 0s 48us/sample - loss: 0.4935 - accuracy: 0.7739 - val_loss: 0.4132 - val_accuracy: 0.8436
    Epoch 265/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.5026 - accuracy: 0.7963 - val_loss: 0.5045 - val_accuracy: 0.7821
    Epoch 266/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4818 - accuracy: 0.7809 - val_loss: 0.4516 - val_accuracy: 0.8268
    Epoch 267/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4742 - accuracy: 0.7963 - val_loss: 0.4106 - val_accuracy: 0.8212
    Epoch 268/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4991 - accuracy: 0.7865 - val_loss: 0.4544 - val_accuracy: 0.8045
    Epoch 269/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4916 - accuracy: 0.7739 - val_loss: 0.5170 - val_accuracy: 0.7542
    Epoch 270/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4884 - accuracy: 0.7837 - val_loss: 0.4322 - val_accuracy: 0.8268
    Epoch 271/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.5098 - accuracy: 0.7837 - val_loss: 0.4182 - val_accuracy: 0.8324
    Epoch 272/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4903 - accuracy: 0.7781 - val_loss: 0.4476 - val_accuracy: 0.8212
    Epoch 273/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4993 - accuracy: 0.7626 - val_loss: 0.4837 - val_accuracy: 0.7877
    Epoch 274/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4809 - accuracy: 0.7963 - val_loss: 0.4423 - val_accuracy: 0.8268
    Epoch 275/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4875 - accuracy: 0.7753 - val_loss: 0.4640 - val_accuracy: 0.8212
    Epoch 276/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4777 - accuracy: 0.8090 - val_loss: 0.4639 - val_accuracy: 0.8101
    Epoch 277/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4919 - accuracy: 0.7823 - val_loss: 0.4515 - val_accuracy: 0.8212
    Epoch 278/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4814 - accuracy: 0.7963 - val_loss: 0.4484 - val_accuracy: 0.8212
    Epoch 279/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4979 - accuracy: 0.7907 - val_loss: 0.5182 - val_accuracy: 0.7486
    Epoch 280/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4893 - accuracy: 0.7837 - val_loss: 0.4320 - val_accuracy: 0.8380
    Epoch 281/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4935 - accuracy: 0.7963 - val_loss: 0.4648 - val_accuracy: 0.7877
    Epoch 282/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4561 - accuracy: 0.7978 - val_loss: 0.4256 - val_accuracy: 0.8268
    Epoch 283/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4826 - accuracy: 0.7823 - val_loss: 0.4615 - val_accuracy: 0.8045
    Epoch 284/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4978 - accuracy: 0.7823 - val_loss: 0.4211 - val_accuracy: 0.8380
    Epoch 285/512
    712/712 [==============================] - 0s 48us/sample - loss: 0.4882 - accuracy: 0.7809 - val_loss: 0.4022 - val_accuracy: 0.8380
    Epoch 286/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4974 - accuracy: 0.7739 - val_loss: 0.4103 - val_accuracy: 0.8492
    Epoch 287/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4734 - accuracy: 0.7893 - val_loss: 0.4222 - val_accuracy: 0.8436
    Epoch 288/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4937 - accuracy: 0.7907 - val_loss: 0.4197 - val_accuracy: 0.8492
    Epoch 289/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4715 - accuracy: 0.7907 - val_loss: 0.4519 - val_accuracy: 0.8212
    Epoch 290/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4991 - accuracy: 0.7935 - val_loss: 0.4178 - val_accuracy: 0.8380
    Epoch 291/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4831 - accuracy: 0.7893 - val_loss: 0.4401 - val_accuracy: 0.8268
    Epoch 292/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4919 - accuracy: 0.7907 - val_loss: 0.4075 - val_accuracy: 0.8436
    Epoch 293/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.4754 - accuracy: 0.7851 - val_loss: 0.4238 - val_accuracy: 0.8492
    Epoch 294/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4868 - accuracy: 0.8020 - val_loss: 0.4345 - val_accuracy: 0.8045
    Epoch 295/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4888 - accuracy: 0.8020 - val_loss: 0.4450 - val_accuracy: 0.8156
    Epoch 296/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4813 - accuracy: 0.7879 - val_loss: 0.4479 - val_accuracy: 0.8212
    Epoch 297/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4710 - accuracy: 0.8034 - val_loss: 0.3997 - val_accuracy: 0.8547
    Epoch 298/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4626 - accuracy: 0.7963 - val_loss: 0.5322 - val_accuracy: 0.7542
    Epoch 299/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4667 - accuracy: 0.7992 - val_loss: 0.5419 - val_accuracy: 0.7263
    Epoch 300/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4925 - accuracy: 0.7935 - val_loss: 0.4731 - val_accuracy: 0.7821
    Epoch 301/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4749 - accuracy: 0.8020 - val_loss: 0.4620 - val_accuracy: 0.8156
    Epoch 302/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4709 - accuracy: 0.7963 - val_loss: 0.4659 - val_accuracy: 0.7877
    Epoch 303/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4783 - accuracy: 0.7837 - val_loss: 0.4243 - val_accuracy: 0.8380
    Epoch 304/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4826 - accuracy: 0.7992 - val_loss: 0.4452 - val_accuracy: 0.8101
    Epoch 305/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4779 - accuracy: 0.7992 - val_loss: 0.5009 - val_accuracy: 0.7821
    Epoch 306/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4881 - accuracy: 0.7879 - val_loss: 0.4594 - val_accuracy: 0.7933
    Epoch 307/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4675 - accuracy: 0.8118 - val_loss: 0.4020 - val_accuracy: 0.8547
    Epoch 308/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4807 - accuracy: 0.7949 - val_loss: 0.3914 - val_accuracy: 0.8436
    Epoch 309/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4515 - accuracy: 0.8034 - val_loss: 0.4219 - val_accuracy: 0.8380
    Epoch 310/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4843 - accuracy: 0.8020 - val_loss: 0.4194 - val_accuracy: 0.8436
    Epoch 311/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4760 - accuracy: 0.7992 - val_loss: 0.4370 - val_accuracy: 0.8101
    Epoch 312/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4860 - accuracy: 0.7893 - val_loss: 0.4106 - val_accuracy: 0.8492
    Epoch 313/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.4559 - accuracy: 0.8062 - val_loss: 0.3977 - val_accuracy: 0.8492
    Epoch 314/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.4677 - accuracy: 0.8048 - val_loss: 0.3899 - val_accuracy: 0.8603
    Epoch 315/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4788 - accuracy: 0.8076 - val_loss: 0.3979 - val_accuracy: 0.8547
    Epoch 316/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4773 - accuracy: 0.7978 - val_loss: 0.4061 - val_accuracy: 0.8436
    Epoch 317/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4715 - accuracy: 0.7949 - val_loss: 0.4123 - val_accuracy: 0.8436
    Epoch 318/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4709 - accuracy: 0.8146 - val_loss: 0.4517 - val_accuracy: 0.8045
    Epoch 319/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4671 - accuracy: 0.8034 - val_loss: 0.4228 - val_accuracy: 0.8380
    Epoch 320/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4897 - accuracy: 0.7978 - val_loss: 0.4209 - val_accuracy: 0.8380
    Epoch 321/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4750 - accuracy: 0.8020 - val_loss: 0.4503 - val_accuracy: 0.8045
    Epoch 322/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4809 - accuracy: 0.7949 - val_loss: 0.3898 - val_accuracy: 0.8547
    Epoch 323/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4826 - accuracy: 0.8034 - val_loss: 0.3847 - val_accuracy: 0.8492
    Epoch 324/512
    712/712 [==============================] - 0s 47us/sample - loss: 0.4679 - accuracy: 0.8090 - val_loss: 0.4286 - val_accuracy: 0.8212
    Epoch 325/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4657 - accuracy: 0.8034 - val_loss: 0.3976 - val_accuracy: 0.8547
    Epoch 326/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4636 - accuracy: 0.8216 - val_loss: 0.3897 - val_accuracy: 0.8547
    Epoch 327/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4819 - accuracy: 0.7992 - val_loss: 0.4180 - val_accuracy: 0.8436
    Epoch 328/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4837 - accuracy: 0.7893 - val_loss: 0.4026 - val_accuracy: 0.8492
    Epoch 329/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4593 - accuracy: 0.8146 - val_loss: 0.4045 - val_accuracy: 0.8436
    Epoch 330/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4710 - accuracy: 0.8020 - val_loss: 0.3893 - val_accuracy: 0.8659
    Epoch 331/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4469 - accuracy: 0.8034 - val_loss: 0.4236 - val_accuracy: 0.8380
    Epoch 332/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4757 - accuracy: 0.7978 - val_loss: 0.4274 - val_accuracy: 0.8380
    Epoch 333/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4681 - accuracy: 0.8132 - val_loss: 0.4058 - val_accuracy: 0.8436
    Epoch 334/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4651 - accuracy: 0.8020 - val_loss: 0.3983 - val_accuracy: 0.8547
    Epoch 335/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4797 - accuracy: 0.8034 - val_loss: 0.4318 - val_accuracy: 0.8268
    Epoch 336/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4824 - accuracy: 0.7907 - val_loss: 0.3906 - val_accuracy: 0.8547
    Epoch 337/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4766 - accuracy: 0.8006 - val_loss: 0.3823 - val_accuracy: 0.8492
    Epoch 338/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4805 - accuracy: 0.7879 - val_loss: 0.3983 - val_accuracy: 0.8436
    Epoch 339/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4695 - accuracy: 0.7992 - val_loss: 0.4235 - val_accuracy: 0.8268
    Epoch 340/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4581 - accuracy: 0.8104 - val_loss: 0.4198 - val_accuracy: 0.8324
    Epoch 341/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4542 - accuracy: 0.8174 - val_loss: 0.4125 - val_accuracy: 0.8436
    Epoch 342/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4678 - accuracy: 0.7992 - val_loss: 0.3785 - val_accuracy: 0.8324
    Epoch 343/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4556 - accuracy: 0.8104 - val_loss: 0.3884 - val_accuracy: 0.8547
    Epoch 344/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4654 - accuracy: 0.8146 - val_loss: 0.3828 - val_accuracy: 0.8603
    Epoch 345/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4634 - accuracy: 0.8146 - val_loss: 0.3803 - val_accuracy: 0.8492
    Epoch 346/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4743 - accuracy: 0.8132 - val_loss: 0.3989 - val_accuracy: 0.8380
    Epoch 347/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.4581 - accuracy: 0.8006 - val_loss: 0.3865 - val_accuracy: 0.8547
    Epoch 348/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4516 - accuracy: 0.8132 - val_loss: 0.3840 - val_accuracy: 0.8492
    Epoch 349/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4684 - accuracy: 0.8146 - val_loss: 0.3775 - val_accuracy: 0.8603
    Epoch 350/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4468 - accuracy: 0.8146 - val_loss: 0.4057 - val_accuracy: 0.8380
    Epoch 351/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4477 - accuracy: 0.8188 - val_loss: 0.3860 - val_accuracy: 0.8492
    Epoch 352/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4554 - accuracy: 0.8020 - val_loss: 0.3859 - val_accuracy: 0.8492
    Epoch 353/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4592 - accuracy: 0.8076 - val_loss: 0.4034 - val_accuracy: 0.8380
    Epoch 354/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4437 - accuracy: 0.8188 - val_loss: 0.4043 - val_accuracy: 0.8380
    Epoch 355/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4626 - accuracy: 0.8132 - val_loss: 0.3969 - val_accuracy: 0.8380
    Epoch 356/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4577 - accuracy: 0.8048 - val_loss: 0.3905 - val_accuracy: 0.8492
    Epoch 357/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4550 - accuracy: 0.8104 - val_loss: 0.3889 - val_accuracy: 0.8436
    Epoch 358/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4450 - accuracy: 0.8160 - val_loss: 0.4011 - val_accuracy: 0.8436
    Epoch 359/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4451 - accuracy: 0.8076 - val_loss: 0.3924 - val_accuracy: 0.8547
    Epoch 360/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4517 - accuracy: 0.8104 - val_loss: 0.3908 - val_accuracy: 0.8547
    Epoch 361/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4545 - accuracy: 0.8202 - val_loss: 0.4181 - val_accuracy: 0.8156
    Epoch 362/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4681 - accuracy: 0.7992 - val_loss: 0.3998 - val_accuracy: 0.8492
    Epoch 363/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4483 - accuracy: 0.8301 - val_loss: 0.4208 - val_accuracy: 0.8324
    Epoch 364/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4655 - accuracy: 0.8104 - val_loss: 0.4012 - val_accuracy: 0.8380
    Epoch 365/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4514 - accuracy: 0.8132 - val_loss: 0.3918 - val_accuracy: 0.8547
    Epoch 366/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4468 - accuracy: 0.8132 - val_loss: 0.3848 - val_accuracy: 0.8659
    Epoch 367/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4417 - accuracy: 0.8146 - val_loss: 0.3830 - val_accuracy: 0.8603
    Epoch 368/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4584 - accuracy: 0.8174 - val_loss: 0.3960 - val_accuracy: 0.8547
    Epoch 369/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.4729 - accuracy: 0.8146 - val_loss: 0.4254 - val_accuracy: 0.8101
    Epoch 370/512
    712/712 [==============================] - 0s 48us/sample - loss: 0.4601 - accuracy: 0.8034 - val_loss: 0.3968 - val_accuracy: 0.8324
    Epoch 371/512
    712/712 [==============================] - 0s 47us/sample - loss: 0.4468 - accuracy: 0.8076 - val_loss: 0.3897 - val_accuracy: 0.8436
    Epoch 372/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4583 - accuracy: 0.8104 - val_loss: 0.3838 - val_accuracy: 0.8547
    Epoch 373/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4503 - accuracy: 0.8118 - val_loss: 0.3866 - val_accuracy: 0.8547
    Epoch 374/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4565 - accuracy: 0.8034 - val_loss: 0.3883 - val_accuracy: 0.8324
    Epoch 375/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4488 - accuracy: 0.8062 - val_loss: 0.3741 - val_accuracy: 0.8492
    Epoch 376/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4484 - accuracy: 0.8216 - val_loss: 0.3993 - val_accuracy: 0.8380
    Epoch 377/512
    712/712 [==============================] - 0s 46us/sample - loss: 0.4425 - accuracy: 0.8188 - val_loss: 0.3814 - val_accuracy: 0.8324
    Epoch 378/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4674 - accuracy: 0.8006 - val_loss: 0.3753 - val_accuracy: 0.8603
    Epoch 379/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4588 - accuracy: 0.8160 - val_loss: 0.3763 - val_accuracy: 0.8603
    Epoch 380/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4495 - accuracy: 0.8090 - val_loss: 0.3883 - val_accuracy: 0.8492
    Epoch 381/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4416 - accuracy: 0.8146 - val_loss: 0.3842 - val_accuracy: 0.8492
    Epoch 382/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4485 - accuracy: 0.8104 - val_loss: 0.3903 - val_accuracy: 0.8492
    Epoch 383/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4579 - accuracy: 0.8006 - val_loss: 0.3700 - val_accuracy: 0.8547
    Epoch 384/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4467 - accuracy: 0.8104 - val_loss: 0.3761 - val_accuracy: 0.8492
    Epoch 385/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4297 - accuracy: 0.8258 - val_loss: 0.3969 - val_accuracy: 0.8380
    Epoch 386/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4530 - accuracy: 0.8034 - val_loss: 0.3944 - val_accuracy: 0.8380
    Epoch 387/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4628 - accuracy: 0.8132 - val_loss: 0.3793 - val_accuracy: 0.8492
    Epoch 388/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4326 - accuracy: 0.8258 - val_loss: 0.4186 - val_accuracy: 0.8324
    Epoch 389/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4525 - accuracy: 0.8146 - val_loss: 0.4503 - val_accuracy: 0.8101
    Epoch 390/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4473 - accuracy: 0.8090 - val_loss: 0.4110 - val_accuracy: 0.8324
    Epoch 391/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4390 - accuracy: 0.8230 - val_loss: 0.3801 - val_accuracy: 0.8492
    Epoch 392/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4472 - accuracy: 0.8132 - val_loss: 0.4090 - val_accuracy: 0.8380
    Epoch 393/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4603 - accuracy: 0.8146 - val_loss: 0.3878 - val_accuracy: 0.8492
    Epoch 394/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4585 - accuracy: 0.8146 - val_loss: 0.4254 - val_accuracy: 0.8156
    Epoch 395/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4508 - accuracy: 0.8258 - val_loss: 0.3931 - val_accuracy: 0.8492
    Epoch 396/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4487 - accuracy: 0.8146 - val_loss: 0.3763 - val_accuracy: 0.8547
    Epoch 397/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4585 - accuracy: 0.8006 - val_loss: 0.3890 - val_accuracy: 0.8492
    Epoch 398/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4511 - accuracy: 0.8174 - val_loss: 0.4312 - val_accuracy: 0.8101
    Epoch 399/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4544 - accuracy: 0.7978 - val_loss: 0.4014 - val_accuracy: 0.8324
    Epoch 400/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4357 - accuracy: 0.8160 - val_loss: 0.4020 - val_accuracy: 0.8156
    Epoch 401/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4484 - accuracy: 0.8146 - val_loss: 0.3677 - val_accuracy: 0.8436
    Epoch 402/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4560 - accuracy: 0.7949 - val_loss: 0.3712 - val_accuracy: 0.8380
    Epoch 403/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4649 - accuracy: 0.8062 - val_loss: 0.3745 - val_accuracy: 0.8547
    Epoch 404/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4428 - accuracy: 0.8132 - val_loss: 0.3843 - val_accuracy: 0.8380
    Epoch 405/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4697 - accuracy: 0.8132 - val_loss: 0.3653 - val_accuracy: 0.8492
    Epoch 406/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4563 - accuracy: 0.8090 - val_loss: 0.3681 - val_accuracy: 0.8547
    Epoch 407/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4466 - accuracy: 0.8076 - val_loss: 0.3806 - val_accuracy: 0.8436
    Epoch 408/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4462 - accuracy: 0.8076 - val_loss: 0.3681 - val_accuracy: 0.8492
    Epoch 409/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4279 - accuracy: 0.8146 - val_loss: 0.3878 - val_accuracy: 0.8436
    Epoch 410/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4558 - accuracy: 0.8174 - val_loss: 0.3799 - val_accuracy: 0.8436
    Epoch 411/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4510 - accuracy: 0.8104 - val_loss: 0.3720 - val_accuracy: 0.8380
    Epoch 412/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4573 - accuracy: 0.8062 - val_loss: 0.3808 - val_accuracy: 0.8380
    Epoch 413/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4501 - accuracy: 0.8132 - val_loss: 0.3972 - val_accuracy: 0.8324
    Epoch 414/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4661 - accuracy: 0.8076 - val_loss: 0.3913 - val_accuracy: 0.8380
    Epoch 415/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4429 - accuracy: 0.8146 - val_loss: 0.3749 - val_accuracy: 0.8436
    Epoch 416/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4622 - accuracy: 0.8062 - val_loss: 0.3840 - val_accuracy: 0.8436
    Epoch 417/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4300 - accuracy: 0.8244 - val_loss: 0.3920 - val_accuracy: 0.8324
    Epoch 418/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4527 - accuracy: 0.8104 - val_loss: 0.3910 - val_accuracy: 0.8380
    Epoch 419/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4286 - accuracy: 0.8315 - val_loss: 0.3913 - val_accuracy: 0.8436
    Epoch 420/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4494 - accuracy: 0.8188 - val_loss: 0.3821 - val_accuracy: 0.8436
    Epoch 421/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4495 - accuracy: 0.8258 - val_loss: 0.3830 - val_accuracy: 0.8436
    Epoch 422/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4404 - accuracy: 0.8244 - val_loss: 0.3707 - val_accuracy: 0.8436
    Epoch 423/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4495 - accuracy: 0.8160 - val_loss: 0.3833 - val_accuracy: 0.8436
    Epoch 424/512
    712/712 [==============================] - 0s 32us/sample - loss: 0.4288 - accuracy: 0.8230 - val_loss: 0.3850 - val_accuracy: 0.8268
    Epoch 425/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4368 - accuracy: 0.8301 - val_loss: 0.3691 - val_accuracy: 0.8380
    Epoch 426/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4319 - accuracy: 0.8174 - val_loss: 0.3857 - val_accuracy: 0.8324
    Epoch 427/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4420 - accuracy: 0.8048 - val_loss: 0.3939 - val_accuracy: 0.8324
    Epoch 428/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4390 - accuracy: 0.8202 - val_loss: 0.3806 - val_accuracy: 0.8436
    Epoch 429/512
    712/712 [==============================] - 0s 47us/sample - loss: 0.4277 - accuracy: 0.8188 - val_loss: 0.3991 - val_accuracy: 0.8380
    Epoch 430/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4370 - accuracy: 0.8160 - val_loss: 0.3810 - val_accuracy: 0.8492
    Epoch 431/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4425 - accuracy: 0.8216 - val_loss: 0.3778 - val_accuracy: 0.8436
    Epoch 432/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4553 - accuracy: 0.8258 - val_loss: 0.3708 - val_accuracy: 0.8436
    Epoch 433/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4266 - accuracy: 0.8230 - val_loss: 0.3737 - val_accuracy: 0.8492
    Epoch 434/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4301 - accuracy: 0.8301 - val_loss: 0.3790 - val_accuracy: 0.8547
    Epoch 435/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4449 - accuracy: 0.8230 - val_loss: 0.3781 - val_accuracy: 0.8380
    Epoch 436/512
    712/712 [==============================] - 0s 47us/sample - loss: 0.4577 - accuracy: 0.8202 - val_loss: 0.3796 - val_accuracy: 0.8492
    Epoch 437/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4428 - accuracy: 0.8174 - val_loss: 0.3825 - val_accuracy: 0.8436
    Epoch 438/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4391 - accuracy: 0.8216 - val_loss: 0.3700 - val_accuracy: 0.8436
    Epoch 439/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4366 - accuracy: 0.8230 - val_loss: 0.3769 - val_accuracy: 0.8492
    Epoch 440/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4294 - accuracy: 0.8287 - val_loss: 0.3670 - val_accuracy: 0.8436
    Epoch 441/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4466 - accuracy: 0.8244 - val_loss: 0.3797 - val_accuracy: 0.8380
    Epoch 442/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4523 - accuracy: 0.8020 - val_loss: 0.3779 - val_accuracy: 0.8380
    Epoch 443/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4439 - accuracy: 0.8146 - val_loss: 0.3800 - val_accuracy: 0.8436
    Epoch 444/512
    712/712 [==============================] - 0s 33us/sample - loss: 0.4463 - accuracy: 0.8230 - val_loss: 0.3916 - val_accuracy: 0.8324
    Epoch 445/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4269 - accuracy: 0.8272 - val_loss: 0.3877 - val_accuracy: 0.8380
    Epoch 446/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4430 - accuracy: 0.8104 - val_loss: 0.3698 - val_accuracy: 0.8436
    Epoch 447/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4501 - accuracy: 0.8034 - val_loss: 0.3789 - val_accuracy: 0.8380
    Epoch 448/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4368 - accuracy: 0.8272 - val_loss: 0.3855 - val_accuracy: 0.8380
    Epoch 449/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4298 - accuracy: 0.8160 - val_loss: 0.3806 - val_accuracy: 0.8436
    Epoch 450/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4256 - accuracy: 0.8202 - val_loss: 0.3713 - val_accuracy: 0.8492
    Epoch 451/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4337 - accuracy: 0.8160 - val_loss: 0.3745 - val_accuracy: 0.8436
    Epoch 452/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4608 - accuracy: 0.8020 - val_loss: 0.3737 - val_accuracy: 0.8436
    Epoch 453/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4606 - accuracy: 0.8174 - val_loss: 0.3746 - val_accuracy: 0.8492
    Epoch 454/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4382 - accuracy: 0.8132 - val_loss: 0.3737 - val_accuracy: 0.8380
    Epoch 455/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4319 - accuracy: 0.8287 - val_loss: 0.3773 - val_accuracy: 0.8436
    Epoch 456/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4557 - accuracy: 0.8174 - val_loss: 0.3740 - val_accuracy: 0.8380
    Epoch 457/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4354 - accuracy: 0.8230 - val_loss: 0.3734 - val_accuracy: 0.8380
    Epoch 458/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4318 - accuracy: 0.8244 - val_loss: 0.3752 - val_accuracy: 0.8324
    Epoch 459/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4239 - accuracy: 0.8301 - val_loss: 0.3769 - val_accuracy: 0.8436
    Epoch 460/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4232 - accuracy: 0.8343 - val_loss: 0.3718 - val_accuracy: 0.8547
    Epoch 461/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4318 - accuracy: 0.8287 - val_loss: 0.3800 - val_accuracy: 0.8436
    Epoch 462/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4414 - accuracy: 0.8272 - val_loss: 0.3741 - val_accuracy: 0.8268
    Epoch 463/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4361 - accuracy: 0.8118 - val_loss: 0.3694 - val_accuracy: 0.8380
    Epoch 464/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4414 - accuracy: 0.8104 - val_loss: 0.3711 - val_accuracy: 0.8380
    Epoch 465/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4612 - accuracy: 0.8118 - val_loss: 0.3869 - val_accuracy: 0.8324
    Epoch 466/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4317 - accuracy: 0.8329 - val_loss: 0.3842 - val_accuracy: 0.8380
    Epoch 467/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4542 - accuracy: 0.8188 - val_loss: 0.3834 - val_accuracy: 0.8436
    Epoch 468/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4442 - accuracy: 0.8174 - val_loss: 0.3827 - val_accuracy: 0.8492
    Epoch 469/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4373 - accuracy: 0.8188 - val_loss: 0.3903 - val_accuracy: 0.8436
    Epoch 470/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4284 - accuracy: 0.8244 - val_loss: 0.3862 - val_accuracy: 0.8380
    Epoch 471/512
    712/712 [==============================] - 0s 49us/sample - loss: 0.4226 - accuracy: 0.8216 - val_loss: 0.3815 - val_accuracy: 0.8380
    Epoch 472/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4146 - accuracy: 0.8230 - val_loss: 0.3839 - val_accuracy: 0.8380
    Epoch 473/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4529 - accuracy: 0.8090 - val_loss: 0.4017 - val_accuracy: 0.8380
    Epoch 474/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4451 - accuracy: 0.8188 - val_loss: 0.4187 - val_accuracy: 0.8045
    Epoch 475/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4307 - accuracy: 0.8146 - val_loss: 0.3889 - val_accuracy: 0.8380
    Epoch 476/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4374 - accuracy: 0.8034 - val_loss: 0.4037 - val_accuracy: 0.8324
    Epoch 477/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4523 - accuracy: 0.8160 - val_loss: 0.3974 - val_accuracy: 0.8268
    Epoch 478/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.3951 - accuracy: 0.8357 - val_loss: 0.4007 - val_accuracy: 0.8268
    Epoch 479/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4231 - accuracy: 0.8301 - val_loss: 0.3954 - val_accuracy: 0.8324
    Epoch 480/512
    712/712 [==============================] - 0s 45us/sample - loss: 0.4314 - accuracy: 0.8216 - val_loss: 0.3845 - val_accuracy: 0.8380
    Epoch 481/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4477 - accuracy: 0.8118 - val_loss: 0.3778 - val_accuracy: 0.8380
    Epoch 482/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4256 - accuracy: 0.8118 - val_loss: 0.3681 - val_accuracy: 0.8324
    Epoch 483/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4192 - accuracy: 0.8357 - val_loss: 0.3804 - val_accuracy: 0.8380
    Epoch 484/512
    712/712 [==============================] - 0s 36us/sample - loss: 0.4278 - accuracy: 0.8329 - val_loss: 0.3888 - val_accuracy: 0.8380
    Epoch 485/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4405 - accuracy: 0.8244 - val_loss: 0.3760 - val_accuracy: 0.8380
    Epoch 486/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4454 - accuracy: 0.8146 - val_loss: 0.3858 - val_accuracy: 0.8324
    Epoch 487/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4513 - accuracy: 0.8174 - val_loss: 0.3779 - val_accuracy: 0.8324
    Epoch 488/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4328 - accuracy: 0.8230 - val_loss: 0.3814 - val_accuracy: 0.8380
    Epoch 489/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4423 - accuracy: 0.8258 - val_loss: 0.3859 - val_accuracy: 0.8268
    Epoch 490/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4218 - accuracy: 0.8258 - val_loss: 0.3814 - val_accuracy: 0.8268
    Epoch 491/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4310 - accuracy: 0.8272 - val_loss: 0.3801 - val_accuracy: 0.8268
    Epoch 492/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4544 - accuracy: 0.8160 - val_loss: 0.3786 - val_accuracy: 0.8324
    Epoch 493/512
    712/712 [==============================] - 0s 40us/sample - loss: 0.4067 - accuracy: 0.8371 - val_loss: 0.3897 - val_accuracy: 0.8324
    Epoch 494/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4410 - accuracy: 0.8287 - val_loss: 0.3778 - val_accuracy: 0.8324
    Epoch 495/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4383 - accuracy: 0.8216 - val_loss: 0.3807 - val_accuracy: 0.8268
    Epoch 496/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4220 - accuracy: 0.8357 - val_loss: 0.3962 - val_accuracy: 0.8380
    Epoch 497/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4313 - accuracy: 0.8230 - val_loss: 0.3820 - val_accuracy: 0.8156
    Epoch 498/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4153 - accuracy: 0.8343 - val_loss: 0.3777 - val_accuracy: 0.8324
    Epoch 499/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4342 - accuracy: 0.8287 - val_loss: 0.3788 - val_accuracy: 0.8268
    Epoch 500/512
    712/712 [==============================] - 0s 41us/sample - loss: 0.4573 - accuracy: 0.8062 - val_loss: 0.3737 - val_accuracy: 0.8268
    Epoch 501/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4102 - accuracy: 0.8329 - val_loss: 0.3690 - val_accuracy: 0.8268
    Epoch 502/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4296 - accuracy: 0.8230 - val_loss: 0.3833 - val_accuracy: 0.8324
    Epoch 503/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4338 - accuracy: 0.8287 - val_loss: 0.3705 - val_accuracy: 0.8268
    Epoch 504/512
    712/712 [==============================] - 0s 39us/sample - loss: 0.4269 - accuracy: 0.8301 - val_loss: 0.3802 - val_accuracy: 0.8380
    Epoch 505/512
    712/712 [==============================] - 0s 35us/sample - loss: 0.4367 - accuracy: 0.8272 - val_loss: 0.3763 - val_accuracy: 0.8324
    Epoch 506/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4264 - accuracy: 0.8216 - val_loss: 0.3866 - val_accuracy: 0.8324
    Epoch 507/512
    712/712 [==============================] - 0s 38us/sample - loss: 0.4108 - accuracy: 0.8216 - val_loss: 0.3781 - val_accuracy: 0.8324
    Epoch 508/512
    712/712 [==============================] - 0s 34us/sample - loss: 0.4356 - accuracy: 0.8118 - val_loss: 0.3900 - val_accuracy: 0.8212
    Epoch 509/512
    712/712 [==============================] - 0s 44us/sample - loss: 0.4333 - accuracy: 0.8146 - val_loss: 0.3889 - val_accuracy: 0.8324
    Epoch 510/512
    712/712 [==============================] - 0s 42us/sample - loss: 0.4334 - accuracy: 0.8230 - val_loss: 0.3807 - val_accuracy: 0.8380
    Epoch 511/512
    712/712 [==============================] - 0s 37us/sample - loss: 0.4298 - accuracy: 0.8244 - val_loss: 0.3720 - val_accuracy: 0.8324
    Epoch 512/512
    712/712 [==============================] - 0s 43us/sample - loss: 0.4514 - accuracy: 0.8132 - val_loss: 0.3885 - val_accuracy: 0.8268


## Plot training statistics


```
import matplotlib.pyplot as plt

def plot_history_info(history):
    # get from variable "history"
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = history.epoch

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
```


```
plot_history_info(training_history)
```


![png](titanic_files/titanic_13_0.png)



![png](titanic_files/titanic_13_1.png)


## Check the final accuracy


```
(test_loss, test_accuracy) = model.evaluate(test_x, test_y, batch_size=1)
print("Test accuracy: {}".format(test_accuracy))
```

    418/418 [==============================] - 1s 2ms/sample - loss: 0.3081 - accuracy: 0.9067
    Test accuracy: 0.9066985845565796



```

```
