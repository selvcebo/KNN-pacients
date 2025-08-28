---
#ENGLISH

# 🧬 Breast Cancer Classification with K-Nearest Neighbors (KNN)
## 📌 Description
This project implements a K-Nearest Neighbors (KNN) classification model to diagnose tumors as benign or malignant using the Breast Cancer Wisconsin dataset. The code is written in Python and uses pandas, numpy, and scikit-learn.
```
├── breast-cancer-wisconsin.csv   # Dataset
├── knn_breast_cancer.py          # Main code
├── requirements.txt              # Dependencies
└── README.md                     # Documentation
```

## ⚙️ Installation
```
git clone git@github.com:selvcebo/KNN-pacients.git
cd ./KNN
pip install -r requirements.txt
```

📊 Code Workflow
### 1️⃣ Reading and Preprocessing the Dataset
Replace "?" values with NaN.

Convert data to numeric type.

Fill missing values with the column mean.

### 2️⃣ Separating Variables
X: feature data.

Y: target data.

### 3️⃣ Train-Test Split
80% training, 20% testing.

Use of random_state for reproducibility.

### 4️⃣ Feature Standardization
Normalize data using StandardScaler.

### 5️⃣ Model Creation
Default KNeighborsClassifier() model.

### 6️⃣ Model Training
Fit the model with the training set.

### 7️⃣ Model Evaluation
Confusion matrix.

Classification report (precision, recall, F1-score).

### 8️⃣ Predictions for New Patients
Generate simulated test data.

Classify using the trained model.

## 📈 Example Output
```
[[107   3]
 [  4  57]]
              precision    recall  f1-score   support

           2       0.96      0.97      0.96       110
           4       0.95      0.93      0.94        61

    accuracy                           0.96       171
   macro avg       0.95      0.95      0.95       171
weighted avg       0.96      0.96      0.96       171
```

## 📦 Dependencies
pandas

numpy

scikit-learn

(See requirements.txt for specific versions)

## 📜 License
Distributed under the MIT license. If you reuse this code, give it a star ⭐ on GitHub.

---

## ✨ Author
Sergio Esteban León Valencia | Fullstack Developer | AI & Machine Learning Enthusiast


---
#INGLÉS

# 🧬 Clasificación del cáncer de mama con K-Vecinos Más Cercanos (KNN)
## 📌 Descripción
Este proyecto implementa un modelo de clasificación de K-Vecinos Más Cercanos (KNN) para diagnosticar tumores benignos o malignos utilizando el conjunto de datos de cáncer de mama de Wisconsin. El código está escrito en Python y utiliza Pandas, Numpy y Scikit-Learn. ```
```
├── breast-cancer-wisconsin.csv # Conjunto de datos
├── knn_breast_cancer.py # Código principal
├── requirements.txt # Dependencias
└── README.md # Documentación
```

## ⚙️ Instalación
```
git clone git@github.com:selvcebo/KNN-pacients.git
cd ./KNN
pip install -r requirements.txt
```

📊 Flujo de trabajo del código
### 1️⃣ Lectura y preprocesamiento del conjunto de datos
Reemplazar los valores "?" por NaN.

Convertir los datos a tipo numérico.

Rellenar los valores faltantes con la media de la columna.

### 2️⃣ Separación de variables
X: datos de características.

Y: datos de destino.

### 3️⃣ División de entrenamiento y prueba
80 % de entrenamiento, 20 % de prueba.

Uso de random_state para la reproducibilidad.

### 4️⃣ Estandarización de características
Normalizar los datos con StandardScaler.

### 5️⃣ Creación del modelo
Modelo KNeighborsClassifier() predeterminado.

### 6️⃣ Entrenamiento del modelo
Ajustar el modelo al conjunto de entrenamiento.

### 7️⃣ Evaluación del modelo
Matriz de confusión.

Informe de clasificación (precisión, recuperación, puntuación F1).

### 8️⃣ Predicciones para nuevos pacientes
Generar datos de prueba simulados.

Clasificar utilizando el modelo entrenado.

## 📈 Ejemplo de salida
```
[[107   3]
 [  4  57]]
              precision    recall  f1-score   support

           2       0.96      0.97      0.96       110
           4       0.95      0.93      0.94        61

    accuracy                           0.96       171
   macro avg       0.95      0.95      0.95       171
weighted avg       0.96      0.96      0.96       171
```

## 📦 Dependencias
pandas

numpy

scikit-learn

(Consultar requirements.txt para versiones específicas)

## 📜 Licencia
Distribuido bajo licencia MIT. Si reutilizas este código, dale una estrella ⭐ en GitHub.

---

## ✨ Autor
Sergio Esteban León Valencia | Desarrollador Fullstack | Entusiasta de IA y Machine Learning 
