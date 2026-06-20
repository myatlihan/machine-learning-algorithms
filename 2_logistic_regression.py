'''
Amaç:
    -UCI Heart Disease veri setini kullanarak kalp hastalığı var/yok binray classification tahmini.
    -Logisitic regression modeli ile sınıflandırma ve modelin doğruluk oranı hesaplanır.  

Dataset:
 https://archive.ics.uci.edu/dataset/45/heart+disease    -Kaynak:
    -Hastalara ait çeşitli medikal ölçümler (yaş, cinsiyet, kan basınvı, kolesterol, vb.)
    -Amacımız target olarak kalp hastalığı olup omadığını tahmin etmek.

workflow:
    1.Gerekli kütüphaneler ve veri setinin yüklenmesi.
    2.Dataframe'in oluşturulması.
    3.Eksik verilerin temizlenmesi.
    4.Feature ve target değişkenlerinin ayrılması.
    5.Eğitim ve tes setinin bölünmesi.
    6.Logistic regression modelinin eğitilmesi.
    7.Modelin accuracy değerinin hesaplanması

Kurulumlar:
pip install ucimlrepo scikit-learn pandas numpy
'''

#Kütüphanelerin içe akatarılması.
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


#Datasetin yüklenmesi
heart_disease = fetch_ucirepo(name= 'Heart Disease')
df = pd.DataFrame(data = heart_disease.data.features)
df['target'] = heart_disease.data.targets


#Eksik verilerin temizlenmesi.
if df.isna().any().any():
    df.dropna(inplace=True)


#Features ve target
X = df.drop('target', axis=1)
y = df['target']


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    shuffle=True,
                                                    random_state=42)

print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')


#Logistic Regression modelinin eğitilmesi
log_Reg = LogisticRegression(penalty='l2', C=1, solver='lbfgs', max_iter=100)
log_Reg.fit(X_train, y_train)
y_pred = log_Reg.predict(X_test)

#Accuracy
def acccuracy(y_pred, y_test):
    true_count = np.sum(y_pred == y_test)
    total_count = len(y_test)
    acccuracy  = round(true_count / total_count * 100,3)
    print(acccuracy)

acccuracy(y_pred, y_test)
