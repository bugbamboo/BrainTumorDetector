import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

lstYes=[]
lstNo=[]
Names1 = ['yes']
Names2=['no']
for name in Names1:
    for dirname in os.listdir(name):
        path = os.path.join(name,dirname)
        img = Image.open(path)
        pixel = img.load()
        lstYes.append(np.array(img).reshape(1,1470000))
    arrYes = np.array(lstYes)
for name in Names2:
    for dirname in os.listdir(name):
        path = os.path.join(name,dirname)
        img = Image.open(path)
        lstNo.append(np.array(img).reshape(1,1470000))
    arrNo = np.array(lstNo)
arrYes=arrYes.reshape(140,1470000)
arrNo=arrNo.reshape(90,1470000)
dfYes = pd.DataFrame(data=arrYes,columns=str(list(range(1,1470001))).split(','))
dfYes['isTumor']=np.ones(140)
dfNo = pd.DataFrame(data=arrNo,columns=str(list(range(1,1470001))).split(','),index =range(140,230))
dfNo['isTumor']=np.zeros(90)
dfAll=pd.concat([dfYes,dfNo])
x=dfAll.drop('isTumor',axis=1).values
y=dfAll['isTumor'].apply(int).values
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
scaler= MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(32,activation = 'sigmoid'))
model.add(Dense(4,activation = 'sigmoid'))
model.add(Dense(2,activation = 'sigmoid'))
model.add(Dense(1))
model.compile(optimizer= 'rmsprop', loss = 'binary_crossentropy',metrics = ['accuracy'])
model.fit(x=X_train, y = y_train,epochs=250)
preds = model.predict(X_test)
