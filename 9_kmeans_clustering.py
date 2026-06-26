'''
Amaç:
    -Bu çalışmada , K-means kümeleme algoritması kullanılarak yapay oluşturulan veriler üzerinde kümeleme (clustering) işlemi yapılmaktadır.
    -K means ile birlikte verileri belirli sayıda k kümeye ayırarak benzer özellikteki noktaları bir araya getirir.

Dataset:
    -Sklearn de bulunan make_blobs fonksiyonu ile 300 örnekten ve 4 merkezden oluşan sentetik veri oluştur.
    -Bu veri 2 boyutlu ve kümeler arasında ayrışma cluster_std parametresiyle kontrol edilir.

Workflow:
    1.Kütüphanlerin import edilmesi.
    2.Sentetik veri oluşturulması.
    3.sentetik verinin görselleştirilmesi.
    4.K-means modelinin tanımlanması ve eğitilmesi.
    5.her noktanın ait olduğu kümenin belirlenmesi.
    6.Kümeleme sonuçlarının ve merkez noktalarının görselleştirilmesi.

Kurulumlar:
pip install scikit-learn matplotlib
'''

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Sentetik datanın oluşturulması.
#n_samples: Örnek sayısı.
#centers: Küme sayısı
#cluster_std: Kümelerin yayılma derecesi
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

#Verinin ilk halinin görselleştirilmesi
plt.figure()
plt.scatter(X[:, 0], X[:,1])
plt.title('Etiketsiz veri')
#plt.show()

#Modelin tanımlanması ve train
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

#Her noktanın ait olduğu kümenin belirlenmesi.
labels = kmeans.labels_
centers = kmeans.cluster_centers_

#Kümeleme sonuçlarının ve merkez noktalarının görselleştirilmesi.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c= labels, cmap='viridis')

plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x',s=150, label='Cluster Centers')
plt.title('K-Means Kümeleme Sonuçları' )
plt.legend()
plt.show()