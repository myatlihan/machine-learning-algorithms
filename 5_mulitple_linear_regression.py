'''
Amaç:
    -Bu çalışmada sklearn kütüphanesinde diabets datasetini kullnarak çok değişkenli lineer regresyon modeli ile kan şekeri tahmini yapacağız.
    -Model performansını değerlendireceğiz.

Dataset:
    -Kaynak: sklearn.datasets
    -İçerik: 442 örnek, 10 medikal özellik (yaş,bmi, kan basıncı, vb.)
    -Hedef: Hastanın ilerleme skoru
Workflow:
    1.Datasetin yüklenmesi.
    2.Feture(X) ve dependent variable(y) tanımla.
    3.Train-test ayrımı
    4.Modelin tanımlanması ve eğitim.
    5.Predict
    6.Model başarısının değerlendirilmesi
    7.Görsellştirme

Kurulumlar:
pip install scikit-learn matplotlib numpy
'''

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np 

#Datasetin yüklenmesi
diabetes = load_diabetes()

#Feture(X) ve dependent variable(y) ayrımı
X = diabetes.data
y = diabetes.target

#Train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#Modelin tanımlanması ve eğitilmesi.
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_preds = linear_model.predict(X_test)

#Modelin  değerlendirilmesi
rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print(f"RMSE: {rmse}")


#Modelin gerçek tahmin ne kadar uzaklşatığını görmek
#Residual errora
errors = y_test - y_preds
plt.figure()
plt.plot(range(len(X_test)), errors)
plt.axhline(0, color='red')
plt.title('Residual Errors')
plt.xlabel('Sample index')
plt.ylabel('Error')
plt.show()