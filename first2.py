# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:48:17 2020

@author: casper
"""

import pandas as pd

import matplotlib.pyplot as plt

satis = pd.read_csv("satislar.csv")


aylar=satis[["Aylar"]]

satislar=satis[["Satislar"]]


satislar2=satis.iloc[:,:1].values
#print(satislar2)
print(aylar)
print(satislar)

    

from sklearn.model_selection import train_test_split  # veribölme

x_train,x_test,y_train,y_test=train_test_split(aylar, satislar,test_size=0.33,random_state=0)

from sklearn.preprocessing import StandardScaler

""""
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
"""
 
#verilerin ölçeklenmesi


from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin= lr.predict(x_train)

#görsel
x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))
 
plt.title("Aylık satış tablasu ") 
plt.xlabel("Aylar")
plt.ylabel("Satışlar")


