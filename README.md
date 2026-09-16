# Spaceship Titanic - Predicción de Transporte de Pasajeros

Proyecto de práctica de Machine Learning basado en la competencia ["Spaceship Titanic"](https://www.kaggle.com/competitions/spaceship-titanic) de Kaggle. El objetivo es predecir si un pasajero fue transportado a una dimensión alterna durante el viaje de la nave, a partir de sus datos personales y de consumo a bordo.

## 🎯 Resultado

- **Score en leaderboard de Kaggle:** 0.80032
- **Posición:** 883 de 8,434 participantes (top ~11%)
- **Modelo:** Random Forest Classifier (optimizado con GridSearchCV)
- **Precisión en validación cruzada:** 79.77%

## 🛠️ Proceso

### 1. Análisis exploratorio (EDA)
- Revisión de nulos por columna.
- Verificación del balance de clases de la variable objetivo (`Transported`): ~50.5% / 49.5%, sin desbalance relevante.

### 2. Limpieza de datos
- `HomePlanet`, `CryoSleep`, `Destination`, `VIP`: nulos rellenados con la **moda**.
- `Age`: nulos rellenados con la **mediana**.
- **Imputación condicional** en las columnas de gasto (`RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`): se comprobó que los pasajeros en `CryoSleep = True` gastan siempre 0 en todos los servicios (verificado con `.describe()`). Por eso, sus nulos se rellenan con 0, mientras que los nulos de pasajeros despiertos se rellenan con la mediana calculada solo entre pasajeros despiertos.

### 3. Feature Engineering
- **Split de `Cabin`** en tres columnas (`deck`, `num`, `side`), separando la cabina en cubierta, número y lado de la nave.
- **`totalSpend`**: suma de las 5 columnas de gasto, como variable agregada adicional.

### 4. Codificación de variables
- One-Hot Encoding para `HomePlanet`, `Destination` y `deck`.
- Codificación binaria simple para `CryoSleep`, `VIP` y `side`.

### 5. Modelado y validación
- Modelo base: Random Forest → 79.47% de precisión en validación simple.
- **Feature importance**: se identificó que `num` (posición de la cabina) resultó ser la variable más influyente del modelo, contradiciendo la hipótesis inicial de que sería poco útil por su alta cardinalidad.
- **Cross-validation (5 folds)**: 78.49% de precisión promedio, con una desviación estándar de 2.19% (resultado estable).
- **GridSearchCV**: búsqueda de hiperparámetros óptimos (`n_estimators`, `max_depth`, `min_samples_leaf`), alcanzando 79.77% de precisión promedio.

### 6. Predicción final
- Reentrenamiento del modelo con los mejores hiperparámetros sobre el 100% de los datos de entrenamiento.
- Aplicación del mismo pipeline de limpieza y transformación sobre `test.csv`, evitando fuga de datos (*data leakage*): todas las estadísticas de imputación (moda, mediana) se calcularon exclusivamente sobre el conjunto de entrenamiento.

## 📁 Contenido del repositorio

- `analisis.ipynb`: notebook completo con el proceso de principio a fin.
- `submission.csv`: archivo de predicciones enviado a Kaggle.

## 🧠 Aprendizajes clave

- La validación cruzada evita conclusiones erróneas por una sola división de datos favorable o desfavorable.
- No se debe descartar una variable por intuición (como se pensó inicialmente con `num`) sin antes verificarlo con `feature_importances_`.
- Cualquier estadística usada para imputar datos debe calcularse únicamente sobre el conjunto de entrenamiento, nunca sobre el conjunto de prueba.
