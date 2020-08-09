"""
importlar
 
*pandas veriler için

axis tablodaki colonları yas fln onları belirtir

"""
import pandas as pd
import numpy as np
import matplotlib as plt 

#kodlar
verilerz = pd.read_csv("veriler.csv")
print(verilerz)

#veri ön işleme

boy = verilerz[["boy"]]     #csv in içindeki boy satırını çağırıyoruz
boykilo = verilerz[["boy","kilo"]] #bundada boy kilo satırını aynı anda çağırıyoruz
print(boy)
print(boykilo)

#eksik veriler
everiler=pd.read_csv("eksikveriler.csv")
print (everiler)
#sci-kit learn
from sklearn.impute import SimpleImputer
 
imputer = SimpleImputer(missing_values=np.nan, strategy = 'mean')
 
Yas = everiler.iloc[:,1:4].values     #iloc pandas kütüphanesinde hangi kolonları almamızı belirlediğimiz fonk.
print(Yas)
 
 
imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas) 
#veri tipi değiştirme
ulke = everiler.iloc[:,0:1].values
print(ulke)
#encoder: katagorik -- numeric
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder() #labelları birebir sayıya çeviriyor
ulke[:,0] = le.fit_transform(ulke[:,0])# 0dan sayı vererek başlatıyor
print(ulke)
ohe = OneHotEncoder(categories='auto')#sayıları colon bazlı çeviriyor.
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)
print(list(range(22)))
#tablo oluşturma data frame kodu
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr", "us"])
print(sonuc)

sonuc1=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo", "yas"])
print(sonuc1)

cinsiyet = verilerz.iloc[:,-1:].values
print(cinsiyet)

sonuc2=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc2)

s=pd.concat([sonuc,sonuc1],axis=1) #data frame birleştirmek concat

print(s)

s1=pd.concat([s,sonuc2],axis=1) #data frame birleştirmek concat

print(s1)

# veribölme

from sklearn.model_selection import train_test_split # veribölme

x_train, x_test, y_train, y_test = train_test_split(s,sonuc2,test_size=0.33,random_state =0) #0,33 alınca örnekler tr ve us olacak fr kullanılmayacak sağlıklınbi öğrenme için her durumdan örnek lazım olduğu için random
# random verinin başarı oranını değiştirir

#normleştirme standartlaştırma



#Konu 3 tahmin
"""
satis = pd.read_csv("satislar.csv")


aylar=satis["Aylar"]

satislar=satis["Satislar"]


#satislar2=satis.iloc[:,:1].values
#print(satislar2)
print(aylar)



from sklearn.model_selection import train_test_split # veribölme

x_train, y_train,x_test , y_test = train_test_split(aylar,satislar,test_size=0.33,random_state =0)



#verilerin ölçeklenmesi


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

"""
  
"""
#nesne tabanlı
 
class insan:
    boy = 190 
    def kosmak(self,b):
       return b + 10
ali=insan()
print (ali.boy)
print (ali.kosmak(90))
"""         
        



