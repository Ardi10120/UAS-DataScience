import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler



# Step 1: Membaca data dari file menggunakan Pandas

dir = '/cleveland.data'
# dir = '/content/drive/MyDrive/Kuliah/Tugas/Data Science/hungarian.data'


# Read lines from the file
with open(dir, encoding='Latin1') as file:
    lines = [line.strip() for line in file]

lines[0:10]

data = itertools.takewhile(
    lambda x: len(x) == 76,
    (' '. join(lines[i:(i + 10)]). split() for i in range (0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df.head()

df = df.apply(pd.to_numeric, errors='coerce')

df.replace(-9, float(0), inplace=True)



df.replace(-9, float('NaN'), inplace=True)


selected_features = df.iloc[:, [2, 3, 8, 9, 11, 15, 18, 31, 37, 39, 40, 43, 50, 57]]

selected_features.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

print(selected_features)


# Membuat heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(selected_features.corr(), annot=True, cmap='inferno', fmt=".2f")

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Pisahkan fitur dan target
X = selected_features.drop('num', axis=1)  # Fitur kecuali kolom target 'num'
y = selected_features['num']  # Kolom target 'num'

# Tentukan urutan yang diinginkan untuk kelas target
kelas_order = sorted(y.unique())  # Urutan "01234"

# Hitung distribusi kelas sebelum SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(y, order=kelas_order)
plt.title('Distribusi Kelas Target Sebelum SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Terapkan SMOTE pada data latih saja
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Hitung distribusi kelas setelah SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(y_train_smote, order=kelas_order)
plt.title('Distribusi Kelas Target Setelah SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()



# Misalkan selected_features adalah DataFrame yang sudah dipersiapkan dan dibersihkan sebelumnya

# Pisahkan fitur dan target
X = selected_features.drop('num', axis=1)  # Fitur kecuali kolom target 'num'
y = selected_features['num']  # Kolom target 'num'

# Split data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

# Evaluasi model Decision Tree
acc_dt = accuracy_score(y_test, y_pred_dt)
print("Akurasi Decision Tree:", acc_dt)
if acc_dt >= 0.7:
    print("Model Decision Tree memenuhi syarat minimal akurasi 70%")
else:
    print("Model Decision Tree tidak memenuhi syarat minimal akurasi 70%")
print("Confusion Matrix Decision Tree:")
cm_dt = confusion_matrix(y_test, y_pred_dt)
print(cm_dt)

# Plot confusion matrix Decision Tree
plt.figure(figsize=(8, 6))
sns.heatmap(cm_dt, annot=True, cmap='Blues', fmt='d', xticklabels=dt_model.classes_, yticklabels=dt_model.classes_)
plt.title('Confusion Matrix - Decision Tree')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Inisialisasi model Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluasi model Random Forest
acc_rf = accuracy_score(y_test, y_pred_rf)
print("\nAkurasi Random Forest:", acc_rf)
if acc_rf >= 0.7:
    print("Model Random Forest memenuhi syarat minimal akurasi 70%")
else:
    print("Model Random Forest tidak memenuhi syarat minimal akurasi 70%")
print("Confusion Matrix Random Forest:")
cm_rf = confusion_matrix(y_test, y_pred_rf)
print(cm_rf)

# Plot confusion matrix Random Forest
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, cmap='Greens', fmt='d', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Inisialisasi model SVM
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluasi model SVM
acc_svm = accuracy_score(y_test, y_pred_svm)
print("\nAkurasi SVM:", acc_svm)
if acc_svm >= 0.7:
    print("Model SVM memenuhi syarat minimal akurasi 70%")
else:
    print("Model SVM tidak memenuhi syarat minimal akurasi 70%")
print("Confusion Matrix SVM:")
cm_svm = confusion_matrix(y_test, y_pred_svm)
print(cm_svm)

# Plot confusion matrix SVM
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, cmap='Oranges', fmt='d', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

