'''
Amaç: 
    -Bu çalışmada, iris veri seti kullanılarak PCA ile veri boyutunu 4'ten 2'ye indirgeme yapılacaktır ve sınıflar 2 boyutlu uzayda görselleştirilecek. 

Dataset:
    -Kaynak: sklearn.datasets iris veri seti
    -İçerik: 150 sample, 4 features (sepal length, sepal width, petal length, petal width)
    -Traget: 3 class (setosa, versicolor, vriginica)

Workflow:
    1.Datasetin yüklenmesi.
    2.feature (X) ve target (y) değişkenlerinin tanımlanması.
    3.PCA ile veri boyutu 2 bileşene indirilecek.
    4.2 boyutlu PCA sonucunun görselleştirilmesi.
    5.Classların renkler ile ayrıştırılması

'''

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()

X= iris.data
y = iris.target


pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

print(X_pca)
for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y==i,0], X_pca[y==i, 1], label=iris.target_names[i])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('İris Dataset (2 Boyutlu)')
plt.legend()
plt.show()
