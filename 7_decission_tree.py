'''
Amaç:
    - Breast Cancer veri seti üzerinde Decision Tree algoritması ile tümör sınıflandırması yapmak.
    - Model performansını accuracy, classification report ve confusion matrix ile değerlendirmek.

Dataset:
    - Kaynak: sklearn.datasets.load_breast_cancer()
    - 569 sample
    - 30 feature
    - 2 class

Workflow:
    1. Datasetin yüklenmesi
    2. Feature ve target ayrımı
    3. Train-test split
    4. Decision Tree modelinin eğitimi
    5. Tahmin
    6. Değerlendirme
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#Datasetin içe aktarılması.
cancer = load_breast_cancer()

#Feature ve target
X = cancer.data
y = cancer.target

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

#Model
model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5,)

#Train
model.fit(X_train, y_train)

#Predict
y_pred =model.predict(X_test)

#Accuracy
accuracy =accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

#Classification Report
print('Classification report:\n')
print(classification_report(y_test,
                            y_pred,
                            target_names=cancer.target_names))
#Confussion Matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

#Tree visualization
plt.figure(figsize=(20,10))

plot_tree(
    model,
    feature_names=cancer.feature_names,
    class_names=cancer.target_names,
    filled=True
)

plt.show()