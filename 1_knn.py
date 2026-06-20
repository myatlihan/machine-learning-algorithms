'''
Amaç:
    -Bu program, göğüs kanseri veri setini kullanarak K-Nearest Neighbors (KNN) algoritmasıyla sınıflandırma yapmayı amaçlamaktadır.
    -Modelin doğruluk oranı hesaplanır, farklı k değerleri için doğruluk analizi yani hiperparametre ayarı yapılır.

Workflow:
    1.Veri setinin yüklenmesi ve incelenmesi
    2.Özellik(feature) ve hedef(target) değişjenlerinin ayrılması.
    3.Train ve test verilerinin ayrılması.
    4.Özelliklerin ölçeklendirilmesi(standardization).
    5.KNN classifierin eğitilmesi ve test edilmesi.
    6.Accuracy ve confusion mantrixin hesaplanması.
    7.Hiperparametre ayarlaması
    8.Sonuçların grafiksel olarak gösterilmesi.
'''
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Veri setinin yüklenmesi ve ilk 3 satırın ekrana yazdırılması
df = pd.read_csv('Dataset/breast_cancer.csv')
print(df.head(3))

#Feature ve target değişkenlerinin ayrılması
X = df.drop('target', axis=1)
y = df['target']

#Train ve test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=42)

#Normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Hiperparametre seçimi 
accuracy_list= []
k_values = range(1, 20)
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k) #KNN model
    knn.fit(X_train_scaled,y_train)
    predicts = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test,predicts)
    accuracy_list.append(accuracy)

#k değerlerimize karşılık gelen accuracy skorlarımızın görselleştirilmesi
plt.plot(k_values, accuracy_list)
plt.xlabel('k values')
plt.ylabel('accuracy')
plt.xticks(range(1,20))
plt.show()

#Hiperparamatre seçimi sonrasında en iyi k değerimizi alıyoruz
best_accuracy = max(accuracy_list)
best_k = accuracy_list.index(best_accuracy) + 1

#hiperparamtre ayarıyla birlikte en iyi en iyi model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

#Confussion matrix
cm = confusion_matrix(y_test, y_pred)


print(f"Best k: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")
print(cm)  

