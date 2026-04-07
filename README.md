# Modelizacion de Datos 2026

Proyecto de Machine Learning aplicado a la prediccion de impago de prestamos (LendingClub dataset, 2007-2017). El objetivo es construir un modelo que, dado un conjunto de variables de un prestamo, prediga si el prestatario pagara o entrara en impago (*Charged Off*).

A lo largo del curso se sigue un enfoque iterativo y progresivo: partimos de los datos en crudo, construimos un modelo base de referencia, y vamos mejorando el pipeline de preprocesamiento y seleccion de variables hasta obtener un modelo mas robusto.

## Estructura del proyecto

```
.
├── 01_analisis_descriptivo.ipynb      # Exploracion y analisis descriptivo
├── 02_train_test_split.ipynb          # Particion de datos train/test
├── 03_modelo_base.ipynb               # Modelo base de referencia (FICO)
├── 04_preprocesamiento.ipynb          # Preprocesamiento manual paso a paso
├── 05_clean_preprocessing.ipynb       # Preprocesamiento refactorizado + filtrado + modelo
├── src/
│   ├── preprocessing/
│   │   └── base_preprocessing.py      # Clase BasePreprocess (fit/transform)
│   └── filtering/
│       └── base_filtering.py          # Clase BaseFiltering (fit/transform)
├── data/                              # Datasets (CSV y Excel de configuracion)
├── pyproject.toml                     # Configuracion del proyecto y dependencias
└── uv.lock                            # Lock de dependencias
```

## Flujo de trabajo

### 1. Particion de datos (`02_train_test_split.ipynb`)

El primer paso es **aislar los datos de test** antes de cualquier analisis, para evitar data leakage.

- Se carga el dataset original (~1.76M prestamos, 151 variables).
- Se filtran solo los prestamos con estado terminal: **Fully Paid** (~80%) y **Charged Off** (~20%).
- Se generan tres conjuntos mediante muestreo estratificado:
  - `df_train_small.csv` -- 80,000 filas, distribucion original (~80/20).
  - `df_train_balanced.csv` -- 80,000 filas, balanceado 50/50 mediante undersampling.
  - `df_test_small.csv` -- 20,000 filas, distribucion original (~80/20).
- El test **no se balancea**: debe reflejar la distribucion real para una evaluacion valida.

> Se trabaja con subconjuntos pequenos para agilizar la experimentacion en clase.

### 2. Analisis descriptivo (`01_analisis_descriptivo.ipynb`)

Con el conjunto de train ya aislado, se realiza la exploracion de datos:

- Inspeccion de dimensiones, tipos de datos y valores nulos.
- Identificacion de 36 variables categoricas y 115 numericas.
- Analisis de cardinalidad de categoricas (ej: `emp_title` con ~37K valores unicos).
- Deteccion de variables con alto porcentaje de nulos (algunas al 100%).
- Visualizaciones: boxplots, densidades (KDE), heatmaps y tablas cruzadas contra `loan_status`.
- Uso de `skrub.TableReport` para profiling automatico.

### 3. Modelo base de referencia (`03_modelo_base.ipynb`)

Se construye un modelo **naive** basado unicamente en el FICO score, para tener una linea base contra la que comparar:

- Se normalizan los rangos FICO (300-850) al intervalo [0, 1].
- Se aplica un umbral de 0.67 (~670 FICO) como punto de corte.
- Metricas obtenidas en test:
  - **Accuracy:** ~72%
  - **Precision (impago):** ~26%
  - **Recall (impago):** ~24%
- Se visualizan la curva ROC y la curva de calibracion.

> Este modelo sirve de referencia: cualquier modelo posterior debe superar estas metricas.

### 4. Preprocesamiento manual (`04_preprocesamiento.ipynb`)

En este notebook se desarrolla **paso a paso** el pipeline de preprocesamiento, explicando cada decision:

**Seleccion de variables:**
- Se usa un fichero Excel (`variables_withoutExperts.xlsx`) que indica las variables candidatas.
- Se eliminan identificadores (`id`, `member_id`), variables con fuga de informacion post-aprobacion (`funded_amnt`), y variables irrelevantes.
- Se reduce de 151 a ~50 variables.

**Tratamiento de nulos (estrategia por umbrales):**
- **>98% nulos** -- se elimina la variable (ej: `sec_app_*`).
- **<10% nulos** -- se imputa con la **mediana** (numericas) o la **moda** (categoricas).
- **10-98% nulos** -- se imputa con **-1** (numericas) o **"DESCONOCIDO"** (categoricas), creando una categoria explicita de dato ausente.

**Variables categoricas:**
- **Baja cardinalidad** (<=50 valores) -- One-Hot Encoding.
- **Alta cardinalidad** (ej: `emp_title`, `desc`) -- Embeddings semanticos con `skrub.TextEncoder` (modelo `e5-small-v2`, 20 componentes).
- **Fechas** (`earliest_cr_line`) -- Extraccion de ano y mes.
- Se elimina `zip_code` por redundancia con `addr_state`.

**Variables numericas:**
- `QuantileTransformer` para transformar a distribucion normal.
- `SquashingScaler` (skrub) para comprimir outliers.

**Ensamblaje final:**
- Se concatenan todas las variables procesadas en una unica matriz de features.
- Se entrena un `RandomForestClassifier` como primera prueba.
- Se aplican las **mismas transformaciones** (ya ajustadas) al conjunto de test.

> Concepto clave: **fit solo en train, transform en train y test**.

### 5. Refactorizacion y filtrado (`05_clean_preprocessing.ipynb`)

El codigo manual del notebook anterior se **refactoriza en clases** dentro de `src/`, y se anade un pipeline de seleccion de variables:

**Preprocesamiento refactorizado** (`BasePreprocess`):
- Se instancia con el fichero Excel y la variable target.
- `.fit(data)` aprende todos los parametros de transformacion.
- `.transform(data)` aplica las transformaciones y devuelve `(X_data, y_data)`.
- Incluye ademas: **PolynomialFeatures** (interacciones de grado 2 entre variables numericas).

**Filtrado de variables** (`BaseFiltering`):
- Pipeline de 3 etapas, tambien con patron fit/transform:
  1. **DropConstantFeatures** -- elimina variables quasi-constantes (tolerancia 0.9).
  2. **DropCorrelatedFeatures** -- elimina variables con correlacion >0.8.
  3. **ProbeFeatureSelection** -- inyecta 10 variables aleatorias como "probes" y elimina las variables reales que no superan el rendimiento del ruido. Usa un RandomForest internamente.

**Modelo:**
- Se entrena un `RandomForestClassifier(n_estimators=100, class_weight='balanced')`.
- `class_weight='balanced'` penaliza mas los errores en la clase minoritaria (impago).
- Evaluacion completa en test: accuracy, precision, recall, F1, AUC-ROC, matriz de confusion y curva ROC.

> En banca, el **recall** sobre impagos es critico: un impago no detectado representa una perdida financiera real.

## Codigo en `src/`

### `src/preprocessing/base_preprocessing.py` -- Clase `BasePreprocess`

Encapsula todo el preprocesamiento en una clase con interfaz `fit` / `transform`:

| Metodo | Descripcion |
|--------|-------------|
| `__init__(var_to_process, target)` | Recibe la ruta al Excel de variables y el nombre de la variable objetivo |
| `fit(data)` | Ajusta todos los transformadores sobre los datos de entrenamiento |
| `transform(data)` | Aplica las transformaciones y devuelve `(X_data, y_data)` |

Etapas internas:
1. Seleccion de variables desde Excel.
2. Tratamiento de nulos (3 umbrales).
3. One-Hot Encoding (categoricas de baja cardinalidad).
4. Extraccion de features temporales.
5. Embeddings de texto (`emp_title`, `desc`).
6. Transformacion de numericas (QuantileTransformer).
7. Features de interaccion (PolynomialFeatures grado 2).

### `src/filtering/base_filtering.py` -- Clase `BaseFiltering`

Pipeline de seleccion de variables en 3 etapas:

| Metodo | Descripcion |
|--------|-------------|
| `__init__(...)` | Configura umbrales y parametros del filtrado |
| `fit(X_data, y_data)` | Ajusta las 3 etapas secuencialmente sobre train |
| `transform(X_data)` | Aplica el filtrado aprendido |
| `print_summary()` | Muestra cuantas variables se eliminaron en cada etapa |

Etapas:
1. Eliminacion de variables constantes/quasi-constantes.
2. Eliminacion de variables altamente correlacionadas.
3. Seleccion por probes (solo se conservan variables mas informativas que ruido aleatorio).

## Dependencias principales

- **pandas** -- Manipulacion de datos.
- **scikit-learn** -- Modelos, metricas, transformadores.
- **feature-engine** -- Seleccion de variables (constantes, correlaciones, probes).
- **skrub** -- TextEncoder (embeddings), SquashingScaler.
- **sentence-transformers** -- Modelo de embeddings `e5-small-v2`.
- **matplotlib / seaborn** -- Visualizaciones.

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd modelizacion_datos_2026

# Instalar dependencias con uv
uv sync
```

## Resumen del flujo completo

```
Datos crudos (1.76M prestamos)
    │
    ▼
[02] Train/Test Split (estratificado)
    │
    ├── train (80K) ──────────────────────────────┐
    │                                              │
    ▼                                              ▼
[01] Analisis descriptivo              [04] Preprocesamiento manual
    │                                      (exploracion paso a paso)
    ▼                                              │
[03] Modelo base (FICO)                            ▼
    (referencia: 72% acc)              [05] Refactorizacion en clases
                                           │
                                           ├── BasePreprocess.fit/transform
                                           ├── BaseFiltering.fit/transform
                                           └── RandomForest + evaluacion
```
