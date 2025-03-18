# modelado.py
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

from utils import get_grid_params

def load_raw_data(filepath):
    """Carga el dataset crudo y convierte la columna 'Date' a datetime.

    Args:
        filepath (str): Ruta al dataset crudo.

    Returns:
        pd.DataFrame: DataFrame con el dataset crudo.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def split_raw_data(df, split_date='2024-01-26'):
    """Separa el dataset crudo en conjuntos de entrenamiento y prueba según una fecha de corte.

    Args:
        df (pd.DataFrame): Dataset crudo.
        split_date (str, optional): Fecha de corte. Por defecto '2024-01-26'.

    Returns:
        tuple: (train_df, test_df)
    """
    df = df.sort_values('Date')
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    return train_df, test_df

def get_features_and_target(df):
    """Extrae las características y el objetivo del DataFrame crudo.

    Genera las siguientes columnas a partir de 'Date':
      - 'Day_of_Month'
      - 'Día_de_la_semana'
      - 'Es_fin_de_semana' (1 si es Saturday o Sunday, 0 de lo contrario)

    Args:
        df (pd.DataFrame): Dataset crudo.

    Returns:
        tuple: (X, y) donde X es el DataFrame de features y y es la columna 'Units_Sold'.
    """
    df['Day_of_Month'] = df['Date'].dt.day
    df['Día_de_la_semana'] = df['Date'].dt.day_name()
    df['Es_fin_de_semana'] = df['Día_de_la_semana'].isin(['Saturday','Sunday']).astype(int)
    X = df[['Store', 'Category', 'Unit_Price', 'Day_of_Month', 'Día_de_la_semana', 'Es_fin_de_semana']]
    y = df['Units_Sold']
    return X, y

def define_features():
    """Define las listas de características numéricas y categóricas para el modelado.

    Returns:
        tuple: (numeric_features, categorical_features)
    """
    numeric_features = ['Unit_Price', 'Day_of_Month']
    categorical_features = ['Store', 'Category', 'Día_de_la_semana', 'Es_fin_de_semana']
    return numeric_features, categorical_features

def configure_pipelines(preprocessor):
    """Configura un diccionario de pipelines para distintos modelos de regresión,
    integrando el preprocesador ya guardado.

    Args:
        preprocessor (ColumnTransformer): Objeto preprocesador cargado.

    Returns:
        dict: Diccionario de pipelines.
    """
    pipelines = {
        'Linear Regression': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ]),
        'Lasso': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Lasso(random_state=42, max_iter=10000))
        ]),
        'Ridge': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', Ridge(random_state=42))
        ]),
        'Decision Tree': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', DecisionTreeRegressor(random_state=42))
        ]),
        'Random Forest': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        'Gradient Boosting': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]),
        'AdaBoost': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', AdaBoostRegressor(random_state=42))
        ])
    }
    return pipelines



def run_grid_search(pipelines, param_grid, X_train, y_train, X_test, y_test, cv=3):
    """Ejecuta GridSearchCV para cada pipeline y devuelve los mejores modelos y sus resultados.

    Args:
        pipelines (dict): Diccionario de pipelines.
        param_grid (dict): Diccionario de grids de hiperparámetros.
        X_train (pd.DataFrame): Datos de entrenamiento (features).
        y_train (pd.Series): Datos de entrenamiento (target).
        X_test (pd.DataFrame): Datos de prueba (features).
        y_test (pd.Series): Datos de prueba (target).
        cv (int, optional): Número de folds para validación cruzada. Por defecto es 3.

    Returns:
        tuple: (best_models, results) donde best_models es un diccionario con los mejores pipelines y results es un diccionario con las métricas.
    """
    best_models = {}
    results = {}
    for model_name, pipeline in pipelines.items():
        print(f"\nEntrenando modelo: {model_name}")
        grid = GridSearchCV(pipeline,
                            param_grid=param_grid[model_name],
                            cv=cv,
                            scoring='neg_mean_absolute_error',
                            n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[model_name] = grid.best_estimator_
        best_params = grid.best_params_
        y_pred = grid.best_estimator_.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        results[model_name] = {'Best Params': best_params, 'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"Mejores parámetros para {model_name}: {best_params}")
        print(f"{model_name}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, R2 = {r2:.2f}")
    return best_models, results

def save_best_models(best_models, save_dir='models/trained_models'):
    """Guarda los modelos entrenados en la carpeta especificada.

    Args:
        best_models (dict): Diccionario con los mejores pipelines.
        save_dir (str, optional): Directorio para guardar los modelos. Por defecto es 'models/trained_models'.
    """
    os.makedirs(save_dir, exist_ok=True)
    for model_name, model in best_models.items():
        model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        print(f"Modelo {model_name} guardado en {model_path}")

def plot_linear_coefficients(model, numeric_features, categorical_features, preprocessor):
    """Extrae y grafica los coeficientes de un modelo lineal.

    Args:
        model (Pipeline): Pipeline que contiene el preprocesador y el regresor lineal.
        numeric_features (list): Lista de características numéricas.
        categorical_features (list): Lista de características categóricas.
        preprocessor (ColumnTransformer): Objeto preprocesador utilizado en el pipeline.
    """
    regressor = model.named_steps['regressor']
    coefs = regressor.coef_
    num_feature_names = numeric_features
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = list(num_feature_names) + list(cat_feature_names)
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    coef_df = coef_df.sort_values(by='Coefficient', key=abs, ascending=False)
    print("Coeficientes del modelo lineal:")
    print(coef_df)
    plt.figure(figsize=(10, 5))
    sns.barplot(data=coef_df, x='Coefficient', y='Feature')
    plt.title("Coeficientes del Modelo Lineal")
    plt.tight_layout()
    plt.savefig('plots/linear_coefficients.png')
    plt.close()
    print("Gráfico de coeficientes guardado en plots/linear_coefficients.png")

def main():
    raw_data_path = 'data/data_raw/sales_data.csv'
    df = load_raw_data(raw_data_path)
    
    #Separamos datos en entrenamiento y prueba
    train_df, test_df = split_raw_data(df, split_date='2024-01-26')
    
    X_train_raw, y_train = get_features_and_target(train_df)
    X_test_raw, y_test = get_features_and_target(test_df)
    
    numeric_features, categorical_features = define_features()
    
    preprocessor = joblib.load('models/preprocessor/preprocessor.pkl')
    
    pipelines = configure_pipelines(preprocessor)
    param_grid = get_grid_params()
    
    best_models, results = run_grid_search(pipelines, param_grid, X_train_raw, y_train, X_test_raw, y_test, cv=3)
    
    results_df = pd.DataFrame(results).T
    print("\nComparación de Modelos:")
    print(results_df)
    
    save_best_models(best_models, save_dir='models/trained_models')
    
    for model_name in ['Linear Regression', 'Lasso', 'Ridge']:
        print(f"\nAnálisis de coeficientes para {model_name}:")
        model = best_models[model_name]
        plot_linear_coefficients(model, numeric_features, categorical_features, preprocessor)

if __name__ == '__main__':
    main()
