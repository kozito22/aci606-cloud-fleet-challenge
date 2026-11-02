# Informe de entrenamiento (v1)

Generado: 2026-11-02T13:41:54+00:00

## 1. EDA

Filas: **48000**

### Distribucion de clases

| clase | muestras | porcentaje |
|---|---|---|
| NORMAL | 4554 | 9.5 |
| WARNING | 7085 | 14.8 |
| CRITICAL | 36361 | 75.8 |

El dataset esta desbalanceado: se usa `class_weight='balanced'` y la metrica principal es **F1 macro**, que da el mismo peso a cada clase.

### Media de cada feature por clase

| clase | speed | battery | motor_temp | vibration | motor_current | obstacle_distance |
|---|---|---|---|---|---|---|
| NORMAL | 0.854 | 80.191 | 52.217 | 0.541 | 4.448 | 1.29 |
| WARNING | 0.809 | 76.715 | 61.31 | 0.954 | 4.877 | 1.3 |
| CRITICAL | 0.747 | 65.32 | 73.822 | 1.525 | 5.466 | 1.313 |

### Informacion mutua con la clase

| feature | informacion_mutua |
|---|---|
| vibration | 0.594 |
| motor_temp | 0.57 |
| motor_current | 0.219 |
| battery | 0.087 |
| speed | 0.04 |
| obstacle_distance | 0.0 |

- `vibration` y `motor_temp` son las senales mas fuertes; `motor_current` y `speed` aportan.
- `obstacle_distance` tiene informacion ~0: es ruido del entorno, no del estado del motor.
- `battery` solo baja con el tiempo y los robots se degradan con el tiempo, asi que su relacion con la clase es un efecto del reloj, no una causa. Una flota nueva llega con bateria alta y puede traer robots CRITICAL desde el inicio.

## 2. Separacion entrenamiento / prueba

GroupShuffleSplit por `fleet_id`: 12 flotas de entrenamiento, 4 flotas de prueba que el modelo nunca ve.
Entrenamiento: 36000 filas, prueba: 12000 filas.

## 3. Comparacion de clasificadores

Validacion cruzada GroupKFold (5) sobre las flotas de entrenamiento + evaluacion en las flotas de prueba. La latencia es la mediana de `predict_proba` para una sola muestra.

| modelo | cv_f1_macro | cv_std | test_f1_macro | latencia_ms |
|---|---|---|---|---|
| baseline_majority [4 features] | 0.2839 | 0.0151 | 0.2878 | 0.019 |
| logistic_regression [4 features] | 0.9203 | 0.0137 | 0.9072 | 0.14 |
| random_forest [4 features] | 0.9196 | 0.0115 | 0.9095 | 40.914 |
| hist_gradient_boosting [4 features] | 0.9199 | 0.0107 | 0.9028 | 3.638 |
| logistic_regression [6 features] | 0.9208 | 0.0126 | 0.9075 | 0.144 |

## 4. Modelo elegido: evidencia en flotas no vistas

F1 macro: **0.9072**

### Matriz de confusion

|  | pred NORMAL | pred WARNING | pred CRITICAL |
|---|---|---|---|
| real NORMAL | 768 | 80 | 0 |
| real WARNING | 160 | 1842 | 34 |
| real CRITICAL | 0 | 261 | 8855 |

### Reporte por clase

```
              precision    recall  f1-score   support

      NORMAL      0.828     0.906     0.865       848
     WARNING      0.844     0.905     0.873      2036
    CRITICAL      0.996     0.971     0.984      9116

    accuracy                          0.955     12000
   macro avg      0.889     0.927     0.907     12000
weighted avg      0.958     0.955     0.956     12000

```

## 5. Justificacion tecnica

- Regla: mayor F1 macro en validacion cruzada por flotas; diferencias < 0.005 se consideran empate y se prefiere el modelo con menos features y menor latencia.
- Elegido: **logistic_regression [4 features]** con F1 macro de prueba **0.907** frente a 0.288 de la linea base de clase mayoritaria.
- La clase depende de una salud interna que el simulador traduce casi linealmente en temperatura, vibracion y corriente; por eso un modelo lineal alcanza el mismo F1 que los ensambles de arboles, sin sobreajustar a flotas concretas.
- Inferencia de una muestra en menos de 1 ms, frente a decenas de ms del Random Forest: importa porque la evaluacion mide p50/p95 de /predict y carga concurrente.
- Usar 4 features en vez de 6 da el mismo F1 y elimina `battery` (efecto del reloj) y `obstacle_distance` (ruido): el modelo depende solo de senales fisicas del motor y generaliza mejor a una flota nueva con otras baterias.
- `predict_proba` de la regresion logistica entrega probabilidades bien calibradas, que se usan como `confidence` en /predict.

## 6. Artefacto

Pipeline (escalado + modelo) reentrenado con todas las flotas y guardado en `models/model.joblib` (scikit-learn 1.9.1). Metadatos en `models/model_meta.json`.
