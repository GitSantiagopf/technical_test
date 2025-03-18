# preprocesamiento.py
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

def load_and_process_data(filepath):
    """Carga el dataset crudo y extrae características temporales.

    Convierte la columna 'Date' a datetime y agrega las siguientes columnas:
      - 'Mes': Mes de la fecha.
      - 'Día_de_la_semana': Nombre del día de la semana.
      - 'Trimestre': Trimestre de la fecha.
      - 'Day_of_Month': Día del mes.
      - 'Es_fin_de_semana': Indicador (1 si es Saturday o Sunday, 0 en caso contrario).

    Args:
        filepath (str): Ruta al archivo CSV de datos crudos.

    Returns:
        pd.DataFrame: DataFrame con las columnas originales y las nuevas variables.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Mes'] = df['Date'].dt.month
    df['Día_de_la_semana'] = df['Date'].dt.day_name()
    df['Day_of_Month'] = df['Date'].dt.day
    df['Es_fin_de_semana'] = df['Día_de_la_semana'].isin(['Saturday', 'Sunday']).astype(int)
    return df

def split_data(df, split_date='2024-01-26'):
    """Separa el DataFrame en conjuntos de entrenamiento y prueba según una fecha de corte.

    Args:
        df (pd.DataFrame): DataFrame completo.
        split_date (str, optional): Fecha límite para el conjunto de entrenamiento.
                                     Los datos a partir de esta fecha se usarán para test.
                                     Por defecto es '2024-01-26'.

    Returns:
        tuple: (train_df, test_df)
    """
    df = df.sort_values('Date')
    train_df = df[df['Date'] < split_date].copy()
    test_df = df[df['Date'] >= split_date].copy()
    return train_df, test_df

def configure_preprocessor(numeric_features, categorical_features):
    """Configura un objeto ColumnTransformer que estandariza las variables numéricas y
    aplica one-hot encoding a las variables categóricas (sin eliminar ninguna categoría)
    y configurado para ignorar categorías desconocidas.

    Args:
        numeric_features (list): Lista de nombres de columnas numéricas.
        categorical_features (list): Lista de nombres de columnas categóricas.

    Returns:
        ColumnTransformer: Objeto configurado para transformar los datos.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    return preprocessor

def save_preprocessor(preprocessor, save_path):
    """Guarda el objeto preprocesador en el disco usando joblib.

    Args:
        preprocessor (ColumnTransformer): Objeto preprocesador a guardar.
        save_path (str): Ruta donde se guardará el objeto.
    """
    joblib.dump(preprocessor, save_path)
    print(f"Preprocesador guardado en '{save_path}'")

def save_preprocessed_data(df_transformed, feature_names, save_path):
    """Guarda el DataFrame transformado en formato CSV.

    Args:
        df_transformed (np.array): Datos transformados por el preprocesador.
        feature_names (list): Nombres de las columnas resultantes.
        save_path (str): Ruta donde se guardará el CSV.
    """
    df_out = pd.DataFrame(df_transformed, columns=feature_names)
    df_out.to_csv(save_path, index=False)
    print(f"Datos preprocesados guardados en '{save_path}'")

def main():
    raw_data_path = 'data/data_raw/sales_data.csv'
    processed_data_dir = 'data/data_processed'
    os.makedirs(processed_data_dir, exist_ok=True)
    
    preprocessor_save_dir = 'models/preprocessor'
    os.makedirs(preprocessor_save_dir, exist_ok=True)
    preprocessor_save_path = os.path.join(preprocessor_save_dir, 'preprocessor.pkl')
    
    df = load_and_process_data(raw_data_path)
    print("Datos cargados y procesados:")
    print(df.head())
    
    full_processed_path = os.path.join(processed_data_dir, 'sales_data_preprocessed_full.csv')
    df.to_csv(full_processed_path, index=False)
    print(f"Dataset completo procesado guardado en '{full_processed_path}'")
    
    train_df, test_df = split_data(df, split_date='2024-01-26')
    print("Datos de entrenamiento:")
    print(train_df.head())
    print("Datos de prueba:")
    print(test_df.head())
    
    numeric_features = ['Unit_Price', 'Day_of_Month']
    categorical_features = ['Store', 'Category', 'Día_de_la_semana', 'Es_fin_de_semana']
    
    preprocessor = configure_preprocessor(numeric_features, categorical_features)
    
    preprocessor.fit(train_df[numeric_features + categorical_features])
    save_preprocessor(preprocessor, preprocessor_save_path)

    X_train_transformed = preprocessor.transform(train_df[numeric_features + categorical_features])
    X_test_transformed = preprocessor.transform(test_df[numeric_features + categorical_features])
    
    num_feature_names = numeric_features
    ohe = preprocessor.named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_features)
    feature_names = list(num_feature_names) + list(cat_feature_names)
    

    train_save_path = os.path.join(processed_data_dir, 'sales_data_train.csv')
    test_save_path = os.path.join(processed_data_dir, 'sales_data_test.csv')
    save_preprocessed_data(X_train_transformed, feature_names, train_save_path)
    save_preprocessed_data(X_test_transformed, feature_names, test_save_path)

if __name__ == '__main__':
    main()
