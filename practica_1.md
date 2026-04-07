# Practica 1: Pipeline completo de Machine Learning para deteccion de impago

## Contexto

En clase hemos construido un pipeline end-to-end de Machine Learning para predecir si un cliente devolvera su prestamo o no (deteccion de impago). Este pipeline incluye:

1. **Preprocesamiento** (`src/preprocessing/base_preprocessing.py`): clase `BasePreprocess` que se encarga de:
   - Tratamiento de valores nulos (eliminacion, imputacion por mediana/moda, valor centinela -1).
   - Encoding de variables categoricas con `OneHotEncoder` (cardinalidad <= 50).
   - Encoding de variables de texto libre con `TextEncoder` (modelo e5-small-v2).
   - Normalizacion de variables numericas con `QuantileTransformer` (distribucion normal).
   - Generacion de features cruzadas con `PolynomialFeatures` (interacciones de grado 2 entre numericas).

2. **Filtrado de features** (`src/filtering/base_filtering.py`): clase `BaseFiltering` que aplica:
   - `DropConstantFeatures`: elimina features cuasi-constantes (tol=0.9).
   - `DropCorrelatedFeatures`: elimina features con correlacion de Pearson > 0.8.
   - `ProbeFeatureSelection`: elimina features menos importantes que ruido aleatorio.

3. **Notebook de referencia** (`04_preprocessing_filtering_model.ipynb`): notebook que encadena preprocesamiento, filtrado, entrenamiento de un Random Forest y evaluacion.

**Importante**: el pipeline de referencia utiliza el fichero `data/variables_withoutExperts.xlsx`, que **excluye** las variables que provienen de evaluaciones de expertos (como el grado de riesgo, el FICO score, la tasa de interes asignada, etc.). El fichero `data/variables_withExperts.xlsx` incluye todas las variables, tambien las de expertos.

---

## Objetivo de la practica

Crear un **pipeline alternativo completo** con vuestras propias decisiones de preprocesamiento, filtrado y modelado, y comparar el rendimiento de **tres familias de modelos** diferentes.

---

## Entrega

La practica se entregara a traves de un **repositorio Git** (GitHub, GitLab o similar). Se debe entregar la **URL del repositorio**. El repositorio debe contener todos los ficheros del proyecto, incluyendo los 4 ficheros indicados a continuacion.

**Requisitos del repositorio:**
- El repositorio debe ser accesible por el profesor (publico o con permisos de lectura).
- El notebook debe estar subido **ya ejecutado** (con las celdas de salida visibles) para poder revisar los resultados sin necesidad de re-ejecutar.

## Ficheros a entregar

### 1. Clase de preprocesamiento: `src/preprocessing/practica1_preprocessing.py`

Crear una clase `Practica1Preprocess` que siga el patron **fit/transform** (igual que `BasePreprocess`) y que incluya las siguientes diferencias respecto a la clase base:

- **Variables**: utilizar el fichero `data/variables_withExperts.xlsx` para incluir tambien las variables de evaluacion de expertos (grade, sub_grade, int_rate, installment, fico_range_low, fico_range_high, etc.).
- **Imputacion de missings**: utilizar al menos un mecanismo de imputacion **diferente** al de la clase base, usando metodos de **scikit-learn** o **feature-engine**. Ejemplos:
  - `sklearn.impute.KNNImputer` (imputacion por vecinos cercanos).
  - `sklearn.impute.IterativeImputer` (imputacion multiple/MICE).
  - `sklearn.impute.SimpleImputer` con estrategia diferente (most_frequent, constant, etc.).
  - Cualquier otro metodo de scikit-learn o feature-engine justificado.
- **Procesamiento de variables categoricas**: utilizar al menos un mecanismo **diferente** a `OneHotEncoder`, usando metodos de **scikit-learn** o **feature-engine**. Ejemplos:
  - `sklearn.preprocessing.TargetEncoder` (encoding basado en la media del target por categoria).
  - `sklearn.preprocessing.OrdinalEncoder` (para variables con orden natural, como grade A-G).
  - `feature_engine.encoding.CountFrequencyEncoder` (encoding por frecuencia).
  - `feature_engine.encoding.MeanEncoder` (encoding por media del target).
  - Cualquier otro metodo de scikit-learn o feature-engine justificado.
- **Procesamiento de variables numericas**: utilizar al menos un mecanismo **diferente** a `QuantileTransformer`, usando metodos de **scikit-learn** o **feature-engine**. Ejemplos:
  - `sklearn.preprocessing.StandardScaler` (estandarizacion z-score).
  - `sklearn.preprocessing.RobustScaler` (escalado basado en mediana e IQR, robusto a outliers).
  - `sklearn.preprocessing.PowerTransformer` (transformacion de Box-Cox o Yeo-Johnson).
  - `sklearn.preprocessing.MinMaxScaler`.
  - Cualquier otro metodo de scikit-learn o feature-engine justificado.
- **Generacion de nuevas features**: utilizar al menos un mecanismo **diferente** a `PolynomialFeatures`. Ejemplos:
  - Ratios financieros: `deuda_total / ingreso_anual`, `cuota / ingreso_mensual`, `balance_revolving / limite_credito`.
  - Agregaciones: combinar fico_range_low y fico_range_high en un fico_medio.
  - Binning: discretizar variables continuas en rangos (ej: tramos de ingreso).
  - Features basadas en dominio bancario.
  - Cualquier otro metodo justificado.

### 2. Clase de filtrado: `src/filtering/practica1_filtering.py`

Crear una clase `Practica1Filtering` que siga el patron **fit/transform** (igual que `BaseFiltering`) y que incluya al menos un mecanismo de filtrado **diferente**. Ejemplos:

- `sklearn.feature_selection.SelectKBest` con test estadistico (chi2, mutual_info_classif, f_classif).
- `sklearn.feature_selection.SelectFromModel` con importancias de un modelo (ej: Lasso, Random Forest).
- `sklearn.feature_selection.VarianceThreshold` (filtrado por varianza).
- `feature_engine.selection.SelectByShuffling` (seleccion por importancia al permutar features).
- `feature_engine.selection.RecursiveFeatureElimination` (RFE) con un modelo ligero.
- `feature_engine.selection.SelectByInformationValue` (seleccion por Information Value).
- Cualquier combinacion de metodos de scikit-learn o feature-engine justificada.

**Nota**: se pueden combinar multiples filtros en secuencia, al igual que hace `BaseFiltering`.

### 3. Notebook: `practica1_notebook.ipynb`

Crear un notebook que ejecute el pipeline completo:

#### 3.1 Preprocesamiento y filtrado
- Instanciar `Practica1Preprocess` con `data/variables_withExperts.xlsx`.
- Hacer `fit()` en train y `transform()` en train y test.
- Instanciar `Practica1Filtering`.
- Hacer `fit()` en train y `transform()` en train y test.
- Mostrar las dimensiones en cada paso.

#### 3.2 Entrenamiento de 3 modelos (uno de cada familia)

Entrenar **tres modelos**, uno de cada familia:

1. **Ensemble de arboles de decision**: `sklearn.ensemble.GradientBoostingClassifier`, `sklearn.ensemble.RandomForestClassifier`, `sklearn.ensemble.AdaBoostClassifier`, o `sklearn.ensemble.HistGradientBoostingClassifier`.
2. **Maquinas de Soporte Vectorial (SVM)**: `sklearn.svm.SVC` con algun kernel (rbf, poly, sigmoid). Usar `probability=True` para poder calcular PR-AUC.
3. **Redes neuronales**: `sklearn.neural_network.MLPClassifier`.

**Notas sobre los modelos**:
- No se pide optimizacion exhaustiva de hiperparametros. Elegir parametros razonables e iterar 2-3 veces como maximo si los resultados iniciales son muy malos.
- Para SVM: tener en cuenta que puede ser lento con muchos datos. Si es necesario, reducir el dataset de entrenamiento con un sample.
- Recordar configurar `random_state` donde sea posible para reproducibilidad.

#### 3.3 Evaluacion

Para **cada uno de los 3 modelos**, calcular y mostrar las siguientes metricas en el **conjunto de test**:

- **Accuracy**.
- **Precision** de la clase de impago (Default = True = clase 1).
- **Recall** de la clase de impago.
- **PR-AUC** (Area Under the Precision-Recall Curve). A diferencia de ROC-AUC, PR-AUC es mas informativa cuando las clases estan desbalanceadas porque se centra en el rendimiento de la clase minoritaria.

```python
from sklearn.metrics import precision_recall_curve, auc

precision_vals, recall_vals, _ = precision_recall_curve(y_test, prob_predicted)
pr_auc = auc(recall_vals, precision_vals)
```

#### 3.4 Comparacion con el modelo base

Incluir una **comparacion explicita** de los resultados de los 3 modelos contra el **modelo base de referencia** construido en clase (`03_modelo_base.ipynb`). Recordar que el modelo base utiliza unicamente el FICO score normalizado con un umbral de 0.67 y obtiene:

- Accuracy: ~72%
- Precision (impago): ~26%
- Recall (impago): ~24%

Se debe:
- Incluir las metricas del modelo base en la **tabla comparativa** junto a las de los 3 modelos (4 filas en total: modelo base + 3 modelos).
- Comentar en que metricas se supera (o no) al modelo base y por cuanto margen.
- Reflexionar: si algun modelo no supera al modelo base en alguna metrica concreta, explicar por que puede estar ocurriendo y que se podria hacer para mejorarlo.

#### 3.5 Comentarios y justificaciones

El notebook debe incluir comentarios que expliquen:
- Por que se eligio cada tecnica de preprocesamiento/filtrado.
- Que parametros se usaron en cada modelo y por que.
- Interpretacion de las metricas obtenidas.

### 4. Fichero de variables: usar `data/variables_withExperts.xlsx`

No es necesario modificar este fichero. Simplemente usarlo como entrada al constructor de `Practica1Preprocess` en lugar de `variables_withoutExperts.xlsx`.

---

## Notas tecnicas

- Los datos de entrenamiento estan en `data/df_train_small.csv` y los de test en `data/df_test_small.csv`.
- La variable target es `loan_status`. La clase positiva (1 = impago) corresponde a `loan_status != 'Fully Paid'`.
- Las clases estan desbalanceadas (~80% Fully Paid vs ~20% Default). Tenerlo en cuenta a la hora de configurar los modelos (`class_weight='balanced'`, `scale_pos_weight`, etc.).
- El patron fit/transform es **fundamental**: todo lo que se aprenda (medianas, encoders, scalers, filtros) debe ajustarse SOLO en train y aplicarse en train y test. Esto evita data leakage.
- **Librerias permitidas**: utilizar exclusivamente **scikit-learn** y **feature-engine** para preprocesamiento, filtrado y modelos. Para visualizacion se puede usar **matplotlib** y **seaborn**. Para manipulacion de datos, **pandas** y **numpy**.

---

## Rubrica de puntuacion (sobre 10)

### Preprocesamiento (3 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Estructura fit/transform correcta | 0.5 | La clase sigue el patron fit/transform sin data leakage. El fit se realiza solo en train. |
| Inclusion de variables de expertos | 0.5 | Se usa `variables_withExperts.xlsx` y se procesan correctamente las variables adicionales (grade, sub_grade, fico, int_rate, etc.). |
| Imputacion de missings alternativa | 0.5 | Se implementa al menos un metodo de imputacion diferente al de la clase base, correctamente aplicado. |
| Procesamiento categoricas alternativo | 0.5 | Se implementa al menos un encoding diferente a OneHotEncoder, correctamente aplicado. |
| Procesamiento numericas alternativo | 0.5 | Se implementa al menos un scaler/transformador diferente a QuantileTransformer, correctamente aplicado. |
| Generacion de nuevas features | 0.5 | Se implementa al menos un mecanismo de generacion de features diferente a PolynomialFeatures (ratios, binning, agregaciones, etc.). |

### Filtrado (1.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Estructura fit/transform correcta | 0.5 | La clase sigue el patron fit/transform sin data leakage. |
| Mecanismo de filtrado alternativo | 0.5 | Se implementa al menos un metodo de filtrado diferente a los de BaseFiltering. |
| Funcionalidad correcta | 0.5 | El filtrado reduce las features de forma efectiva y no genera errores en train/test. |

### Modelos y evaluacion (4.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Ensemble de arboles | 0.75 | Se entrena correctamente un modelo de tipo ensemble de scikit-learn (GradientBoosting, RandomForest, AdaBoost, HistGradientBoosting). |
| SVM con kernel | 0.75 | Se entrena correctamente un `sklearn.svm.SVC` con kernel (rbf, poly, etc.) y `probability=True`. |
| Red neuronal | 0.75 | Se entrena correctamente un `sklearn.neural_network.MLPClassifier`. |
| Metricas completas | 0.75 | Se calculan accuracy, precision, recall y PR-AUC para los 3 modelos en test. |
| Tabla comparativa con modelo base | 0.5 | Se presenta una tabla resumen con las metricas de los 3 modelos **y el modelo base** (4 filas). |
| Comparacion con modelo base | 0.5 | Se comenta en que metricas se supera al modelo base, por cuanto margen, y se reflexiona sobre posibles motivos si algun modelo no lo supera. |
| Clase positiva correcta | 0.5 | La clase 1 (positiva) es el impago (Default = True = loan_status != 'Fully Paid'). |

### Calidad del notebook y entrega (1 punto)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Comentarios y justificaciones | 0.25 | Se explican las decisiones tomadas (por que cada tecnica, por que cada parametro). |
| Interpretacion de resultados | 0.25 | Se interpretan las metricas: que modelo funciona mejor, por que, que implican los resultados en contexto bancario. |
| Repositorio Git y codigo ejecutable | 0.5 | Se entrega un repositorio Git accesible con README, el notebook se ejecuta sin errores y esta subido ya ejecutado con las salidas visibles. El codigo es legible y organizado. |

---

## Fecha de entrega

**Por determinar.**

---

## Recursos utiles

- Documentacion de [feature-engine](https://feature-engine.trainindata.com/) para seleccion y transformacion de features.
- Documentacion de [scikit-learn](https://scikit-learn.org/stable/) para modelos, metricas y preprocesamiento.
- Notebook de referencia: `04_preprocessing_filtering_model.ipynb`.
- Clases de referencia: `src/preprocessing/base_preprocessing.py` y `src/filtering/base_filtering.py`.
