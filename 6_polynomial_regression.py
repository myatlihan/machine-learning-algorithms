'''
Amaç:
    -Bu çalışmada sklearn kütüphanesinde diabets datasetini kullnarak çok değişkenli polinom regresyon modeli ile kan şekeri tahmini yapacağız.
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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#Datasetin içe aktarılması
diabetes =load_diabetes()

#Feature ve target değişkenlerinin ayrılması
X=diabetes.data
y = diabetes.target


#Featurelerin polinom özelliklere dönüşümü



#Train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)


#Modelin tanımlanması ve train
def model(degree:int):
    poly = PolynomialFeatures(degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train_poly,y_train)

    train_preds = linear_model.predict(X_train_poly)
    test_preds = linear_model.predict(X_test_poly)
    return train_preds, test_preds



for degree in range(1, 6):

    train_preds, test_preds = model(degree)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

    print(
        f"Degree={degree} | "
        f"Train RMSE={train_rmse:.2f} | "
        f"Test RMSE={test_rmse:.2f}"
    )
