# Predicción de diabetes con Machine Learning y redes neuronales

Caso de estudio de clasificación binaria en Python: análisis exploratorio, limpieza de datos, modelos tradicionales (regresión logística y árbol de decisión) y una red neuronal artificial para estimar si un paciente pertenece al grupo con diabetes.

**Autor:** Andres Hinojosa
**Contexto:** Parcial de la asignatura Análisis de Datos — UNICIENCIA

## Dataset

[Diabetes Dataset — Kaggle](https://www.kaggle.com/datasets/datascikhan/diabetes) (`diabetes_large_dataset.csv`): 100.000 registros y 209 columnas con variables demográficas, medidas corporales, laboratorios, antecedentes, hábitos y entorno.

El CSV no se incluye en el repositorio. Descárgalo desde Kaggle y súbelo cuando el notebook lo pida.

## Preguntas que responde

1. ¿Qué características presenta la población registrada?
2. ¿Existen diferencias entre los grupos con y sin diabetes?
3. ¿Qué variables tienen mayor relación con la variable objetivo?
4. ¿Qué modelo de clasificación ofrece mejores resultados?
5. ¿Una red neuronal es competitiva frente a un modelo tradicional?

## Contenido del notebook

| Bloque | Tema |
|---|---|
| 1 | Carga e inspección de datos, diccionario de las 209 variables y función de resumen de calidad |
| 2 | Limpieza: selección de variables sin fuga de información, recuperación del objetivo, valores inválidos, atípicos (IQR) y codificación |
| 3 | Análisis exploratorio: distribución del objetivo, histogramas, diagramas de caja, mapa de calor y dispersión |
| 4 | Modelos tradicionales: regresión logística y árbol de decisión con división 80/20 estratificada |
| 5 | Red neuronal (Keras), curvas de pérdida y ajuste del umbral de decisión |
| 6 | Respuestas e informe ejecutivo |

## Decisiones clave

- **Variable objetivo:** `diabetes_status` (1 = con diabetes, 0 = sin diabetes). Los valores faltantes se recuperaron con `diabetes_type` cuando era posible, quedando 99.958 pacientes.
- **Fuga de información:** se descartaron columnas que solo existen después del diagnóstico (medicamentos, complicaciones, puntajes de riesgo). Se usaron 14 predictoras conocidas antes del diagnóstico: edad, IMC, cintura, glucosa en ayunas, HbA1c, presión sistólica y diastólica, HDL, triglicéridos, sexo, antecedente familiar, hipertensión, actividad física y tabaquismo.
- **Valores inválidos:** se marcaron como faltantes en lugar de eliminar filas.
- **Sin fuga en el preprocesamiento:** la imputación por mediana y el escalado se ajustan solo con el conjunto de entrenamiento.
- **Métrica principal:** recall y F1 de la clase con diabetes, porque con un 28 % de casos positivos el accuracy es engañoso (predecir siempre "sin diabetes" ya da 72 %).

## Resultados

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Regresión logística | 0.829 | 0.788 | 0.532 | 0.635 |
| Árbol de decisión | 0.806 | 0.794 | 0.414 | 0.544 |
| Red neuronal (umbral 0.5) | 0.829 | 0.815 | 0.503 | 0.622 |
| Red neuronal (umbral 0.3) | 0.810 | 0.679 | 0.609 | 0.642 |

La regresión logística es el mejor modelo tradicional. La red neuronal queda prácticamente empatada con ella y no muestra sobreajuste; bajar el umbral a 0.3 sube el recall de 0.50 a 0.61 a cambio de más falsas alarmas, un intercambio razonable en un contexto de prevención. La variable más relacionada con el objetivo es la glucosa en ayunas (r = 0.42), seguida de IMC, HbA1c, presión sistólica y circunferencia de cintura.

## Limitaciones

Los datos parecen sintéticos: la HbA1c casi nunca supera 6.5 %, hay miles de glucosas fijadas en 60 mg/dl, columnas redundantes y etiquetas truncadas. La imputación por mediana reduce la variabilidad real y no hay información temporal sobre qué ocurrió antes del diagnóstico. Los resultados no deben usarse con fines clínicos.

## Cómo ejecutarlo

1. Abre `prediccion_diabetes.ipynb` en [Google Colab](https://colab.research.google.com/) (Archivo → Subir notebook, o directamente desde GitHub).
2. Descarga `diabetes_large_dataset.csv` desde Kaggle.
3. Ejecuta todas las celdas; en el bloque 1.B aparecerá un botón para subir el CSV.

El notebook usa una semilla fija (42) para que los resultados sean reproducibles.

## Tecnologías

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · TensorFlow/Keras · Google Colab
