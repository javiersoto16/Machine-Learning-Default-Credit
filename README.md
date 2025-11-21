# 📄 Modelo Predictivo de Riesgo de Impago

# Descripción del proyecto
Construir un modelo predictivo capaz de identificar a los clientes con riesgo de impago (clase 1) en un contexto financiero. Se utiliza el dataset de crédito de Taiwán (https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), que contiene información demográfica, histórica de pagos, montos facturados y pagos realizados por los clientes.

---
# Objetivo principal del proyecto
El objetivo es detectar correctamente a los clientes de clase 1, es decir, aquellos con riesgo real de impago.
Por ello, la métrica prioritaria no es la accuracy, sino:
Recall de la clase 1 porque dejar pasar un cliente de riesgo (falso negativo) puede generar pérdidas económicas importantes.

---
# Estructura del dataset
El dataset incluye 30,000 registros y 23 variables explicativas, además de la variable objetivo RISK:
Demografía: SEX, EDUCATION, MARRIAGE, AGE
Historial de pagos: PAY_0 a PAY_6
Montos de facturación: BILL_AMT1 a BILL_AMT6
Pagos realizados: PAY_AMT1 a PAY_AMT6
Variable objetivo:
RISK = 1 → cliente con riesgo (impago)
RISK = 0 → cliente sin riesgo

---
# Preprocesamiento realizado
Exploración inicial del dataset
Identificación del desbalance de clases
Separación de variables
Heatmap de correlación para detectar multicolinealidad
Escalado estándar para algoritmos sensibles a la magnitud
Separación en train/test
Aplicación de SMOTE para balancear las clases en entrenamiento
Preparación de X_train, X_test, y_train y y_test

---
# Modelos entrenados
Se entrenaron diversos modelos para comparar su rendimiento:
Modelos base: KNN/ Regresión Logística/ Árbol de Decisión.
Modelos ensemble: Random Forest/AdaBoost/Gradient Boosting.

Cada modelo fue evaluado principalmente según:
Recall (Clase 1) → métrica principal
Precision
F1-score
Matriz de confusión

---
# Visualizaciones incluidas
Gráfico de barras del desbalance de clases
Heatmap de correlaciones con anotaciones

---
# Implementación final y aplicación interactiva
Tras evaluar los modelos, el Gradient Boosting fue seleccionado como el modelo final debido a su mejor rendimiento en recall para la clase 1, crítico para minimizar los falsos negativos en detección de clientes con riesgo de impago.

---
# Aplicación Streamlit
Se desarrolló un app.py para permitir la interacción directa con el modelo:
Entrada manual de datos: Los usuarios pueden introducir los valores de las variables mediante desplegables y campos numéricos, simulando la información que un banco podría registrar para un cliente.
Carga de CSV: Se puede subir un archivo CSV con la misma estructura que el dataset original. La aplicación permite visualizar los datos cargados, aplicar filtros por columnas y revisar la estructura antes de generar predicciones. El modelo calcula la probabilidad de riesgo y la clasificación final (RISK = 0 o 1) para cada registro.
Descarga de resultados: Los usuarios pueden descargar un CSV con las predicciones y probabilidades, listo para análisis o integración en sistemas internos.
Caja de sugerencias y mejora: La última pestaña de la aplicación permite: escribir sugerencias o comentarios sobre el funcionamiento de la app o el modelo, elegir  de una lista interactiva la mejora que considere más importante para optimizar el modelo.
