'''
Amaç:
    -Bu çalışmada, rastgele oluşturulmuş veriler üzerinden lineer regresyon uygulama
    -Amacımız doğrusal ilişkiy modellemek ve tahmin doğrusunu göselleştirmek
    -y = 3 + 4x şeklinde bir veri oluşturmak

Dataset:
    -100 tane rastgele X değeri oluşturmak. (0-1)
    -Gerçek modelimiz: y = 3 + 4x + hata payı
    _Hedefimiz ise a0 = 3 ve a1 = 4 katsayılarını öğrenmek. 

Workflow:
    1.Rastgele veri oluştur.
    2.Linear regression modelinin tanimlanması ve eğitilmesi.
    3.Katsayıların hesaplanması.
    4.Gerçek ve tahmin edilen değerlerin görselleşştirilmesi 

Kurulumlar:
pip install scikit-learn matplotlib numpy
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

#Dataset oluşturma
X =np.random.rand(100,1)
y = 3 + 4*X + np.random.rand(100,1) # y = 3 + 4x + bias


#Modelin oluşturulması ve eğitilmesi.
linear_model = LinearRegression()
linear_model.fit(X,y)

#Predict
y_preds = linear_model.predict(X)

#Gerçek değerler ve tahminlerimiz.
plt.figure()
plt.scatter(X, y, color='red', label='Actual data')
plt.plot(X, y_preds, label= 'Predicted data')
plt.legend()
plt.show()

#R2 score
r2 = r2_score(y, y_preds)
print(f'R2 Score: {r2}')


#Değişken katsayıları
intercept = linear_model.intercept_ # Modelin öğrendiği sabit terim (β0)
coef = linear_model.coef_ # Modelin öğrendiği eğim katsayısı (β1)
df = pd.DataFrame(data=[[intercept, coef]], columns=['intercept', 'coef'])
print(df)