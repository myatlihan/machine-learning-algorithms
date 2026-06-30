'''
Amaç: 
    -Bu çalışmada mnist dataseeti üzerinde LDA uygulyarak 784 boyuttan 2 boyuta indirgemek.
    -Rakam sınıflarının iki boyutlu düzlemde aytımını gözlemek

Dataset:
    -MNIST - openml
    -70.000 sample, her sample 28x28 piksel boyutlarında, yani 784 feature
    -Target: 0-9 arasındaki rakamlar.

Workflow:
    1.MNIST datsetin yüklenmesi.
    2.X ve y değişkenlerinin tanımlanması
    3.LDA modelinin oluşturulması.
    4.Datanın 2 bileşene ayrılması.
    5.2 boyutlu LDA sonucunun görselleştirilmesi.

kurulumlar:
pip install scikit-learn matplotlib openml
'''

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

#Datasetin yüklenmesi.
mnist = fetch_openml('mnist_784', version=1)

#X (feature) ve y (label) değişknelerinin ayrılması. 
X = mnist.data
y = mnist.target.astype(int)

#LDA modelinin oluşturulması
lda = LinearDiscriminantAnalysis(n_components=2)

#Verinin 2 boyuta indirgenmesi
X_lda = lda.fit_transform(X, y)

#LDA sonucunun görselleştirilmesi.
plt.figure()
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10')
plt.title('MNIS-LDA 2D Transformation')
plt.xlabel('LDA1')
plt.ylabel('LDA2')
plt.colorbar(label='Digit class')
plt.show()