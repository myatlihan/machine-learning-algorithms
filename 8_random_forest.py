'''
Amaç:
    - Breast Cancer veri seti üzerinde Random Forest algoritması ile tümör sınıflandırması yapmak.
    - Model performansını accuracy, classification report ve confusion matrix ile değerlendirmek.

Dataset:
    - sklearn.datasets.load_breast_cancer
    - 569 sample
    - 30 feature
    - Binary classification

Workflow:
    1. Dataset yükleme
    2. Feature / target ayrımı
    3. Train-test split
    4. Random Forest model eğitimi
    5. Tahmin
    6. Değerlendirme

Kurulumlar:
pip install scikit-learn pandas matplotlib seaborn
'''

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# 1. Dataset

cancer = load_breast_cancer()

X = cancer.data
y = cancer.target


# 2. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

# Training
model.fit(X_train, y_train)


# 4. Prediction
y_pred = model.predict(X_test)


# 5. Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cancer.target_names,
    yticklabels=cancer.target_names
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 6. Feature Importance
feature_importance = model.feature_importances_

feat_df = pd.DataFrame({
    "Feature": cancer.feature_names,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:\n")
print(feat_df.head(10))