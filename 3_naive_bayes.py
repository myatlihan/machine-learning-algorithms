'''
Amaç: 
    -Bu çalışmada, sklearn kütüphanesinden Iris  dataset kullanılarak çiçek tür sınıflandırma yapılacaktır.
    -Naive bayes algoritması ile model eğitilir, classification report hesaplanır.

Dataset:
    -sklearn.datasets içinde hazır iris veri seti var.
    -İçerik:150 sample var ve her biri 4 adet feature içerir.
    -Target 3 sınıf içerir

Workflow:
    1.Kütüohanelerin import edilmesi ve datasetin yüklenmesi.
    2.Feature ve target değişkeninin tanımlanması.
    3.Train test ayrımı
    4.Naive bayes modelinin eğitilmesi.
    5.Predict
    6.Model performansının değerlendirilmesi.

Kurulumlar:
pip install scikit-learn
'''
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

#Datasetin yüklenmesi.
iris = load_iris()

#Fetaures ve targetin tanımlnması
X = iris.data
y = iris.target

#train test ayrımı 

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
#Modelin tanımlanması 
model = GaussianNB()

#Modelin eğitilmesi
model.fit(X_train,y_train)

#Model tahminlerinin alınması
y_preds = model.predict(X_test)

#Model değerlendirme 
cr = classification_report(y_test,y_preds,target_names=iris.target_names)
print(cr)