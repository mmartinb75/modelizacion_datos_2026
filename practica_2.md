# Practica 2: Optimizacion conjunta calibracion-discriminacion, incertidumbre y servicio del modelo

## Contexto

En clase hemos seguido construyendo el pipeline end-to-end de Machine Learning para deteccion de impago. Sobre la base de Practica 1 (preprocesamiento + filtrado + modelos), hemos visto en los siguientes notebooks de referencia:

1. **Optimizacion bayesiana de hiperparametros** (`11_optuna_tpe.ipynb`): uso de **Optuna** con sampler **TPE** y `MedianPruner` para tunear LightGBM, XGBoost y CatBoost minimizando **Brier score** sobre un split interno train/val. Incluye early stopping con metrica custom y comparacion contra los baselines del notebook 07.

2. **Calibracion de probabilidades** (`10_calibracion.ipynb`): diagnostico con `reliability diagram`, `Brier score` y `Expected Calibration Error (ECE)`, y aplicacion de **Platt scaling** (sigmoid) e **isotonic regression** mediante `sklearn.calibration.CalibratedClassifierCV`. Reflexion sobre **cuando calibrar** y **cuando no merece la pena** (modelos ya calibrados, datasets pequenos, etc.).

3. **Conformal Prediction** (`09_conformal_prediction.ipynb`): construccion de **intervalos de prediccion** con cobertura garantizada (Inductive Conformal Prediction, split-conformal). Permite cuantificar la **incertidumbre** de cada prediccion y derivar las decisiones inciertas a un humano (o a un agente especializado).

**Punto de partida comun**: esta practica **debe partir de los artefactos de preprocesamiento y filtrado generados en clase**, sin volver a "fittear" preprocesador ni filtros (esa fase es la mas costosa computacionalmente y ya se hizo). Concretamente se reutiliza:

- **Objetos proces** (objetos subidos en modulo `practica 2`):
  - `base_pre.pkl` (objeto de preprocesamiento)
  - `base_filter.pkl` (objeto de filtrado)

- **Objetos fitteados** (necesarios sobre todo en el Repo 2, para procesar inputs crudos que llegan a la API, objetos subidos en modulo `practica 2`):
  - `preprocessor.pkl`: el pipeline de **preprocesado ya fitteado** sobre train.
  - `filter.pkl`: el pipeline de **filtrado de features ya fitteado** sobre train.

> **Importante**: estos `.pkl` **no son datos**, son **objetos** (instancias de las clases de Practica 1) con sus parametros aprendidos y sus metodos incluidos (`transform`, etc.). Es decir, una vez cargados con `joblib.load(...)` se invocan directamente como `preprocessor.transform(df_raw)` o `feature_filter.transform(df_pre)` para procesar **datos nuevos** (por ejemplo, el payload que llegue a la API). Esto evita repetir el `fit` en cada arranque.

Estos artefactos se generan al final del notebook `06_clean_preprocessing_and_filtering_model.ipynb` (que ya aplica el preprocesado sobre `data/variables_withExperts.xlsx` y el filtrado de features). **No se debe re-preprocesar ni re-filtrar en esta practica**: se cargan los pickles y se trabaja sobre ellos. También se adjuntan en el contenido extra de la práctica, dentro del módulo practica 2.

---

## Objetivo de la practica

Construir, sobre el pipeline ya conocido, un **flujo completo de modelado y servicio** que incluya:

1. **Optimizacion automatica** de hiperparametros con Optuna usando una metrica que **mejore la calibracion sin degradar la discriminacion**.
2. **Calibracion** de las probabilidades del mejor modelo, justificando si es necesario o no.
3. **Cuantificacion de incertidumbre** y politica de **derivacion a un agente** cuando la prediccion no es fiable.
4. **API local** con dos servicios (subida de modelo + prediccion) lista para servir el modelo.

---

## Entrega

La practica se entregara como **dos repositorios Git independientes** (GitHub). Se debe entregar **una URL por repositorio**. **No es necesario desplegar nada en la nube**: basta con que todo se pueda ejecutar en local siguiendo el `README` de cada repo.

| Repo | Contenido | Stack principal |
|------|-----------|-----------------|
| **Repo 1 - Modelado** | Notebook con Optuna, calibracion, medida de incertidumbre y persistencia del modelo final (`.pkl`) | Python + scikit-learn + Optuna |
| **Repo 2 - API** | API REST que carga el `.pkl` y sirve predicciones con politica de derivacion | FastAPI + uvicorn |

**Requisitos comunes a los dos repos:**
- Accesibles por el profesor (publicos o con permisos de lectura).
- `README` con instrucciones claras de instalacion y ejecucion en local.
- Se entrega cada repo con su artefacto final: el `.pkl` del modelo (Repo 1) y la API arrancable (Repo 2).
- En el Repo 1, los notebooks deben subirse **ya ejecutados** (con las celdas de salida visibles).

## Contenido de cada repositorio

### Repo 1 - Modelado: notebook `practica2_notebook.ipynb`

Notebook que ejecute, de principio a fin, los pasos de modelado, calibracion e incertidumbre, y guarde al final el artefacto del modelo que consumira la API. Estructura sugerida:

#### 1.1 Optimizacion con Optuna: calibracion **y** discriminacion (variacion al notebook 11)

El notebook 11 optimizaba **Brier score** (que penaliza calibracion + resolucion juntas) y los notebooks anteriores (07) miraban solo AUC. Aqui queremos optimizar una metrica que **mejore la calibracion sin degradar la discriminacion**.

- **Cargar directamente** los pickles `preprocessor.pkl` y `filter.pkl` (mismos artefactos que el notebook 11). No re-preprocesar ni re-filtrar.
- Optimizar  **dos modelos** (de LightGBM, XGBoost, CatBoost o equivalente) con **Optuna + TPE** y aplicar las siguientes **variaciones** respecto al notebook 11:
  - **Metrica objetivo**: usar una metrica que capture ambas dimensiones. Elegir **una** de las siguientes opciones (justificar la eleccion):
    1. **Log Loss** (`sklearn.metrics.log_loss`): proper scoring rule, descomponible en `resolution + reliability`. Penaliza tanto la mala discriminacion como la mala calibracion. Es la opcion mas sencilla y la recomendada por defecto.
    2. **Metrica combinada** custom, ej. `AUC - lambda * ECE` o `MCC - lambda * Brier`, con `lambda` razonado.
  - **Sampler/pruner**: cambiar al menos uno de los siguientes parametros respecto al notebook 11. Por ejemplo: `HyperbandPruner` o `SuccessiveHalvingPruner` en lugar de `MedianPruner`, o cambiar `n_startup_trials` / `multivariate` del `TPESampler`. Justificar el cambio.
  - **Espacio de busqueda**: anadir o ampliar al menos un hiperparametro respecto al notebook 11 (ej. `min_child_weight` en XGBoost, `feature_fraction_bynode` en LightGBM, `border_count` en CatBoost).
  - **Comparacion balanceado vs no balanceado**: para cada modelo, lanzar el `study` **dos veces**:
    1. Con tratamiento de desbalanceo (`class_weight='balanced'`, `scale_pos_weight=neg/pos` o `auto_class_weights='Balanced'`).
    2. **Sin** tratamiento de desbalanceo.
    
    Comparar el `best_value` (la metrica elegida) y, sobre todo, las metricas finales en test. Comentar que opcion gana en calibracion (Brier/ECE) y que ocurre con discriminacion (AUC/MCC). **Importante**: el balanceo de clases sesga las probabilidades; comentar el efecto sobre la calibracion.

- Para cada modelo final, calcular en test y mostrar en una tabla: **Accuracy, Precision, Recall, F1, MCC, ROC-AUC, PR-AUC, Log Loss, Brier, ECE**.
- Eleccion del **modelo ganador** que pasa a la siguiente fase, justificada por la metrica elegida.

#### 1.2 Calibracion (variacion al notebook 10)


- **Decision**: tomar de forma **explicita** la decision de calibrar o no, **justificada** con los numeros del diagnostico.
- **Si se decide calibrar**: aplicar **un metodo fiable** (`sigmoid` / `isotonic` u otro justificado) y verificar que la calibracion mejora (`ECE`, `Brier`) **sin degradar** la discriminacion (`AUC`, `MCC`).
- **Si se decide NO calibrar**: explicar por que el modelo no necesita calibracion, argumentado con datos.

#### 1.3 Medida de incertidumbre y derivacion a un agente

Hasta aqui tenemos una **probabilidad puntual** `p` para cada cliente. Pero esa probabilidad puede ser, ella misma, **incierta**: dos clientes pueden tener `p = 0.55` y, sin embargo, en uno el modelo esta razonablemente seguro de que la probabilidad real esta cerca de 0.55 y en el otro no tiene ni idea (podria estar en cualquier sitio entre 0.4 y 0.7). Queremos detectar este segundo caso y **derivar a un agente humano**.

- **Pregunta abierta a responder en el notebook**:
  > *"Tengo una probabilidad puntual de mi modelo. ¿Que necesito para medir la incertidumbre de esa probabilidad y poder derivar a un agente cuando esa incertidumbre sea alta? ¿Me sirve la calibracion clasica (sigmoid/isotonic)? ¿Que estoy obteniendo realmente con cada metodo?"*

  Argumentar la respuesta y elegir un metodo que permita obtener, para cada prediccion, un **intervalo de probabilidad** `[p_low, p_high]` (la anchura de ese intervalo sera la medida de incertidumbre). Justificar la eleccion del metodo.

- **Implementacion**:
  - Aplicar el metodo elegido al modelo final. Para cada sample de test obtener `p_low` y `p_high`.
  - Reportar la **distribucion de la anchura** `p_high - p_low` sobre test (histograma, percentiles).

- **Politica de derivacion**:
  > Si `|p_high - p_low| > 0.2` -> `decision = "agent"`. En caso contrario -> `decision = "auto"`.

  Es decir, solo dejamos que el modelo decida cuando el intervalo de probabilidad es **estrecho** (modelo seguro de su propia probabilidad).

- **Reflexion adicional**: bajo esta politica, ¿hace falta calibrar puntualmente con sigmoid/isotonic, o el propio metodo de intervalo elegido ya garantiza calibracion sobre los casos que se quedan en `auto`? Razonar.

- **Metricas con derivacion** sobre test:
  - `% de casos derivados al agente`.
  - Accuracy / Precision / Recall **solo sobre los casos auto-decididos**.
  - Comparar con las metricas globales del paso 1.1 para mostrar el trade-off cobertura/precision.

#### 1.4 Persistencia del modelo

Guardar como ultimo paso del notebook:
- `practica2_model.pkl`: pipeline final (modelo + calibrador si lo hay + objeto de intervalo). Es **un objeto fitteado** con sus metodos (`predict`, `predict_proba`, y el metodo que devuelva el intervalo `[p_low, p_high]`), no datos.


El Repo 2 (API) consumira este ficheros junto con los `preprocessor.pkl` y `filter.pkl` heredados de Practica 1, encadenando: `df_raw -> preprocessor.transform -> filter.transform -> practica2_model.predict_proba`.


### Repo 2 - API local con FastAPI

Repositorio independiente con una API REST en **FastAPI**, ejecutable en local con `uv` y `uvicorn`. **No depende del codigo del Repo 1**: solo consume los `.pkl` y los `.json`. Concretamente carga **tres objetos fitteados** y los encadena en cada prediccion:

1. `preprocessor.pkl` (heredado de Practica 1) -> `preprocessor.transform(df_raw)`.
2. `filter.pkl` (heredado de Practica 1) -> `filter.transform(df_pre)`.
3. `practica2_model.pkl` (generado en el Repo 1) -> `model.predict_proba(...)` y metodo de intervalo.

**Dos endpoints:**

1. **`POST /model/upload`**: sube un nuevo artefacto del modelo al servicio.
   - Acepta el `.pkl` como `multipart/form-data` (o una ruta local).
   - Lo carga, valida que tiene los metodos esperados y lo deja activo en memoria.
   - Devuelve la version y un timestamp.

2. **`POST /predict`**: devuelve la **probabilidad de impago con intervalo de incertidumbre** y la decision.
   - Recibe un JSON con las features **crudas** del cliente (las mismas columnas que `data/variables_withExperts.xlsx`, sin preprocesar).
   - Internamente aplica `preprocessor.transform` -> `filter.transform` -> `model.predict_proba` + metodo de intervalo.
   - Devuelve:
     ```json
     {
       "p_default": 0.31,
       "p_low": 0.22,
       "p_high": 0.41,
       "decision": "auto" | "agent",
       "reason": "p_high - p_low > 0.2"
     }
     ```
   - Regla: `decision = "agent"` si `p_high - p_low > 0.2`. En caso contrario, `decision = "auto"`.

**Requisitos tecnicos:**
- Stack fijo: **FastAPI + uvicorn**, gestionado con `uv`. Se arranca en local con:
  ```bash
  uv run uvicorn api.main:app --reload --port 8080
  ```
- `README` con ese comando + ejemplos de invocacion con `curl` para los dos endpoints + ejemplo de payload.
- Aprovechar la doc automatica de FastAPI en `http://localhost:8080/docs` (Swagger UI) como evidencia de que la API funciona.
- `Dockerfile` **opcional** (suma puntos pero no es obligatorio).

---

## Notas tecnicas

- Repo 1 y Repo 2 usan **`uv`** como gestor de entorno y dependencias (igual que el repo de clase). Las dependencias nuevas (`optuna`, `fastapi`, `uvicorn`, `python-multipart` para el upload, etc.) se anaden con `uv add <paquete>` y quedan reflejadas en `pyproject.toml` / `uv.lock`.
- **Log Loss** esta disponible como `sklearn.metrics.log_loss`. **MCC** como `sklearn.metrics.matthews_corrcoef`. **ECE** se calcula a partir del reliability diagram (binning).

- En la API, **no exponer datos sensibles** (no loguear el payload completo en produccion). En local pueden loguearse para debug.

---

## Rubrica de puntuacion (sobre 10)

### Repo 1 - Optimizacion con Optuna (2.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Metrica que combina calibracion y discriminacion | 0.75 | Se optimiza Log Loss, multi-objetivo o metrica combinada justificada (no solo Brier ni solo AUC). |
| Variacion respecto al notebook 11 | 0.5 | Al menos un cambio justificado en sampler, pruner o espacio de busqueda. |
| Comparacion balanceado vs no balanceado | 0.5 | Cada modelo se entrena en las dos modalidades y se comparan resultados. |
| Tabla final y eleccion del ganador | 0.75 | Tabla con metricas (Accuracy, Precision, Recall, F1, MCC, AUC, PR-AUC, Log Loss, Brier, ECE) y eleccion justificada del modelo. |

### Repo 1 - Calibracion (1.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Diagnostico (reliability diagram + ECE + Brier + Log Loss) | 0.5 | Se diagnostica la calibracion del modelo ganador con metricas y grafico. |
| Decision argumentada (calibrar o no) | 0.5 | Se decide calibrar o no con justificacion numerica, no por defecto. |
| Aplicacion correcta (si calibra) o argumentacion (si no) | 0.5 | Si calibra, aplica un metodo fiable y verifica que no degrada AUC/MCC. Si no, justifica solidamente. |

### Repo 1 - Incertidumbre y derivacion a un agente (2.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Identificacion del metodo necesario | 0.75 | Se responde la pregunta abierta y se justifica el metodo elegido para obtener un intervalo de probabilidad por prediccion. |
| Implementacion del intervalo `[p_low, p_high]` | 0.5 | Se aplica el metodo elegido y se reporta la distribucion de la anchura sobre test. |
| Politica de derivacion (anchura > 0.2) | 0.75 | Regla implementada; se reporta `%` derivado + metricas restringidas a casos auto vs metricas globales. |
| Persistencia del artefacto | 0.5 | El notebook guarda `practica2_model.pkl` y `feature_schema.json`. |

### Repo 2 - API con FastAPI (2.5 puntos)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Endpoint `/model/upload` | 0.75 | Sube y carga un nuevo `.pkl`, lo valida y lo deja activo. |
| Endpoint `/predict` con intervalo | 1.25 | Devuelve `p_default`, `p_low`, `p_high` y `decision` segun la regla `p_high - p_low > 0.2`. |
| Ejecucion local + Swagger | 0.5 | `uv run uvicorn ...` arranca, `/docs` accesible, `README` con `curl` de ejemplo. |

### Calidad y entrega (1 punto)

| Criterio | Puntos | Descripcion |
|----------|--------|-------------|
| Dos URLs entregadas y accesibles | 0.25 | Repo 1 y Repo 2 entregados como repositorios independientes. |
| README en cada repo | 0.5 | Cada repo tiene `README` propio con instrucciones de ejecucion en local. |
| Reproducibilidad | 0.25 | `random_state` fijado, dependencias congeladas (`uv.lock`), instrucciones claras para reproducir Repo 1 + Repo 2. |

---

## Fecha de entrega

**Por determinar.**

---

## Recursos utiles

- Notebooks de referencia: `09_conformal_prediction.ipynb`, `10_calibracion.ipynb`, `11_optuna_tpe.ipynb`.
- Documentacion de [Optuna](https://optuna.readthedocs.io/) (samplers TPE, pruners Median/Hyperband, multi-objective).
- `sklearn.calibration.CalibratedClassifierCV`, `sklearn.calibration.calibration_curve`.
- `sklearn.metrics.log_loss`, `sklearn.metrics.brier_score_loss`, `sklearn.metrics.matthews_corrcoef`.
- Documentacion de [FastAPI](https://fastapi.tiangolo.com/) y [uvicorn](https://www.uvicorn.org/).
