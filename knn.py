# Contenido
# Content
# Lectura del dataset.
# Reading the dataset.
# Identificar la data de características  X  y la data objetivo  y.
# Identify the feature data X and the target data y.
# Obtener la data de entrenamiento  Xtrain,  ytrain  y la data de prueba  Xtest,  ytest.
# Obtain the training data Xtrain, ytrain and the test data Xtest, ytest.
# Estandarizar las características  Xtrain  y  Xtest.
# Standardize the features Xtrain and Xtest.
# Generar el modelo de clasificación KNN.
# Generate the KNN classification model.
# Entrenamos el modelo KNN con la información  Xtrain  y  ytrain.
# Train the KNN model with the information Xtrain and ytrain.
# Hacemos pruebas de funcionalidad del modelo KNN con  Xtest  y  ytest.
# Test the functionality of the KNN model with Xtest and ytest.
# Hacemos clasificaciones con nuevos pacientes.
# Make classifications with new patients.

# Importar las librerías necesarias
# Import the necessary libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Lectura del dataset
# 1. Reading the dataset

df = pd.read_csv("breast-cancer-wisconsin.csv")
df.head()

df["Núcleos desnudos"] = df["Núcleos desnudos"].replace("?", np.nan)
df["Núcleos desnudos"] = pd.to_numeric(df["Núcleos desnudos"])
avg_nucleos = df["Núcleos desnudos"].mean()
df["Núcleos desnudos"] = df["Núcleos desnudos"].fillna(avg_nucleos)
df.info()

# 2. Identificar la data de características  X  y la data objetivo Y.
# 2. Identify the feature data X and the target data Y.

Y = df["Clase"].copy()
X = df.drop(["Clase", "Id"], axis=1).copy()
Y.loc[:5]

# 3. Obtener la data de entrenamiento  Xtrain,  ytrain  y la data de prueba  Xtest,  ytest.
# 3. Obtain the training data Xtrain, ytrain and the test data Xtest, ytest.

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X.shape, X_train.shape, X_test.shape, Y.shape, Y_train.shape, Y_test.shape

# 4. Estandarizar las características  Xtrain  y  Xtest.
# 4. Standardize the features Xtrain and Xtest.

mi_reescalador = StandardScaler()
X_train_scaled = mi_reescalador.fit_transform(X_train)
X_test_scaled = mi_reescalador.transform(X_test)

# 5. Generar el modelo de clasificación KNN.
# 5. Generate the KNN classification model.

mi_knn = KNeighborsClassifier()

# 6. Entrenamos el modelo KNN con la información  Xtrain  y  ytrain.
# 6. Train the KNN model with the information Xtrain and ytrain.

mi_knn.fit(X_train_scaled, Y_train)

# 7. Hacemos pruebas de funcionalidad del modelo KNN con  Xtest  y  ytest.
# 7. Test the functionality of the KNN model with Xtest and ytest.

Y_pred_knn = mi_knn.predict(X_test_scaled)
print(confusion_matrix(Y_test, Y_pred_knn))
print(classification_report(Y_test, Y_pred_knn))

# 8. Hacemos clasificaciones con nuevos pacientes.
# 8. Make classifications with new patients.

pacientes_nuevos = {
    'Grosor del grumo': [4.41, 4.41-2.81, 4.41+2.81],
    'Uniformidad del tamaño de las células': [3.134478, 3.134478 - 3.051459, 3.134478 + 3.051459],
    'Uniformidad de la forma de las células': [3.207439, 3.207439 - 2.971913, 3.207439 + 2.971913],
    'Adhesión marginal': [2.806867, 2.806867 - 2.855379, 2.806867 + 2.855379],
    'Tamaño de una sola célula epitelial': [3.216023, 3.216023 - 2.214300, 3.216023 + 2.214300],
    'Núcleos desnudos': [3.544656, 3.544656 - 3.601852, 3.544656 + 3.601852],
    'Cromatina suave': [3.437768, 3.437768 - 2.438364, 3.437768 + 2.438364],
    'Nucléolos normales': [2.866953, 2.866953 - 3.053634, 2.866953 + 3.053634],
    'Mitosis': [1.589413, 1.589413 - 1.715078, 1.589413 + 1.715078],
}

pacientes_nuevos = pd.DataFrame(pacientes_nuevos)
pacientes_nuevos

pacientes_nuevos_scaled = mi_reescalador.transform(pacientes_nuevos)
predicciones_nuevos_pacientes = mi_knn.predict(pacientes_nuevos_scaled)
print("Predicciones para los nuevos pacientes:", predicciones_nuevos_pacientes)