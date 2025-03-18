# Prueba técnica

Este repositorio contiene la solución a una prueba técnica para profesionales de datos, la cual se centra en la construcción de un modelo predictivo para estimar el número de unidades vendidas (Units_Sold) en una cadena minorista. Se incluyen análisis exploratorio de datos (EDA), preprocesamiento de datos y modelado predictivo.

## Descripción
El objetivo de este proyecto es desarrollar un modelo predictivo basado en datos históricos de ventas, que incluya información sobre la fecha, tienda, categoría, unidades vendidas y precio unitario. Se realizó un análisis exploratorio (EDA) para identificar patrones, tendencias y relaciones en el dataset. Posteriormente, se preprocesaron los datos mediante técnicas de normalización y codificación, y se entrenaron diversos modelos de regresión (incluyendo Linear Regression, Lasso, Ridge, Decision Tree, Random Forest, Gradient Boosting y AdaBoost) con su respectivo tuning de hiperparámetros.

## Estructura del proyecto
```bash
.
├── data/
│   ├── data_raw/               # Dataset original
│   └── data_processed/         # Datos preprocesados (train, test y dataset completo)
│       ├── sales_data_preprocessed_full.csv
│       ├── sales_data_train.csv
│       └── sales_data_test.csv
├── models/
│   ├── preprocessor/           # Objeto preprocesador guardado (preprocessor.pkl)
│   └── trained_models/         # Modelos entrenados
│       ├── Linear_Regression.pkl
│       ├── Lasso.pkl
│       ├── Ridge.pkl
│       ├── Decision_Tree.pkl
│       ├── Random_Forest.pkl
│       ├── Gradient_Boosting.pkl
│       └── AdaBoost.pkl
├── notebooks/                  # Notebook de exploración y desarrollo
│   └── exploracion.ipynb
├── reports/                    # Informes finales en PDF
│   ├── EDA.pdf
│   ├── Preprocess.pdf
│   ├── Modeling.pdf
│   └── Informe final.pdf
├── src/
│   ├── eda.py                  # Módulo de Análisis Exploratorio de Datos
│   ├── preprocesamiento.py     # Módulo de Preprocesamiento de Datos
│   ├── utils.py                # Módulo de funciones de apoyo
│   └── modelado.py             # Módulo de Modelado y Evaluación
├── plots/                      # Gráficos generados por el EDA
│   ├── boxplots.png
│   ├── histograms.png
│   ├── income_by_store.png
│   ├── income_by_category.png
│   ├── time_series.png
│   ├── weekday_analysis.png
│   ├── weekday_vs_weekend.png
│   └── weekday_category_comparison.png
├── README.md
└── requirements.txt            # Dependencias del proyecto
```

## Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## Instrucciones de uso

**1. Análisis Exploratorio de Datos (EDA)**
Para ejecutar el módulo de EDA y generar los gráficos, navega a la carpeta src y ejecuta:

```bash
python eda.py
```

Los gráficos se guardarán en la carpeta plots.


**2. Preprocesamiento de Datos**
Para ejecutar el proceso de preprocesamiento, que incluye la carga, transformación, división (split) y guardado del preprocesador y de los datasets preprocesados, ejecuta:

```bash
python preprocesamiento.py
```

Los datos preprocesados se guardarán en data/data_processed.
El objeto preprocesador se guardará en models/preprocessor/preprocessor.pkl.

**3. Modelado y Evaluación**
Para entrenar y evaluar los modelos predictivos, ejecuta:

```bash
python modelado.py

```

Este módulo realiza:

La carga del dataset crudo y la extracción del target.
El uso del preprocesador guardado para transformar las features.
La configuración de pipelines para diversos modelos de regresión.
La optimización de hiperparámetros mediante GridSearchCV.
La evaluación de modelos con métricas (MAE, RMSE, R²) y el análisis interpretativo (por ejemplo, coeficientes para modelos lineales).
La generación y guardado de los modelos entrenados en models/trained_models.
