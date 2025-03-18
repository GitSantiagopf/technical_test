import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOTS_DIR = '.\plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(filepath):
    """
    Carga el dataset desde un archivo CSV, convierte la columna 'Date' a datetime 
    y extrae algunas características temporales.

    Args:
        filepath (str): Ruta al archivo CSV.

    Returns:
        pd.DataFrame: DataFrame con la columna 'Date' en formato datetime y 
        las columnas 'Mes', 'Día_de_la_semana' y 'Trimestre' agregadas.
    """
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Mes'] = df['Date'].dt.month
    df['Día_de_la_semana'] = df['Date'].dt.day_name()
    df['Trimestre'] = df['Date'].dt.quarter
    df['Day_of_Month'] = df['Date'].dt.day  
    df['Es_fin_de_semana'] = df['Día_de_la_semana'].isin(['Saturday', 'Sunday']).astype(int)
    return df


def descriptive_statistics(df):
    """
    Calcula y muestra las estadísticas descriptivas para 'Units_Sold' y 'Unit_Price'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.

    Returns:
        pd.DataFrame: Estadísticas descriptivas de 'Units_Sold' y 'Unit_Price'.
    """
    stats = df[['Units_Sold', 'Unit_Price']].describe().T
    print("Estadísticas Descriptivas:")
    print(stats)
    return stats

def check_missing_values(df):
    """
    Revisa y muestra la cantidad de valores nulos en cada columna.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.

    Returns:
        pd.Series: Serie con el número de valores nulos por columna.
    """
    missing = df.isnull().sum()
    print("Valores nulos por columna:")
    print(missing)
    return missing

def frequency_analysis(df):
    """
    Imprime la frecuencia de valores en las columnas 'Store' y 'Category'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    print("Frecuencia de tiendas:")
    print(df['Store'].value_counts())
    print("\nFrecuencia de categorías:")
    print(df['Category'].value_counts())

def plot_boxplots(df):
    """
    Genera y guarda boxplots para 'Units_Sold' y 'Unit_Price'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.boxplot(df['Units_Sold'], vert=True, patch_artist=True)
    plt.title("Distribución de Units_Sold")
    plt.ylabel("Units_Sold")
    plt.subplot(1, 2, 2)
    plt.boxplot(df['Unit_Price'], vert=True, patch_artist=True)
    plt.title("Distribución de Unit_Price")
    plt.ylabel("Unit_Price")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'boxplots.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Boxplots guardados en {plot_path}")

def plot_histograms(df):
    """
    Genera y guarda histogramas para 'Units_Sold' y 'Unit_Price'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df['Units_Sold'], bins=10, edgecolor='black')
    plt.title("Histograma de Units_Sold")
    plt.xlabel("Units_Sold")
    plt.ylabel("Frecuencia")
    plt.subplot(1, 2, 2)
    plt.hist(df['Unit_Price'], bins=10, edgecolor='black')
    plt.title("Histograma de Unit_Price")
    plt.xlabel("Unit_Price")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'histograms.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Histogramas guardados en {plot_path}")

def plot_income_analysis(df):
    """
    Calcula la columna 'Ingresos' y genera gráficos de barras guardados que 
    muestran los ingresos totales por 'Store' y 'Category'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    df['Ingresos'] = df['Units_Sold'] * df['Unit_Price']
    
    plt.figure(figsize=(10, 6))
    df.groupby('Store')['Ingresos'].sum().plot(kind='barh')
    plt.title("Ingresos Totales por Tienda")
    plt.ylabel("Store")
    plt.xlabel("Ingresos")
    plot_path = os.path.join(PLOTS_DIR, 'income_by_store.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Ingresos por tienda guardados en {plot_path}")

    plt.figure(figsize=(10, 6))
    df.groupby('Category')['Ingresos'].sum().plot(kind='barh')
    plt.title("Ingresos Totales por Categoría")
    plt.ylabel("Category")
    plt.xlabel("Ingresos")
    plot_path = os.path.join(PLOTS_DIR, 'income_by_category.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Ingresos por categoría guardados en {plot_path}")

def plot_correlation_analysis(df):
    """
    Genera un heatmap de correlación para variables numéricas y binarias.
    
    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    df_corr = df[['Units_Sold', 'Unit_Price', 'Day_of_Month', 'Es_fin_de_semana']]
    
    corr_matrix = df_corr.corr()
    print(corr_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de Correlación (Variables Numéricas y Binarias)")
    plot_path = os.path.join(PLOTS_DIR, 'correlation_heatmap.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Heatmap de correlación guardado en {plot_path}")

def plot_time_series(df):
    """
    Genera y guarda gráficos de series de tiempo para 'Units_Sold' e 'Ingresos'.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df.groupby('Date')['Units_Sold'].sum().plot(kind='line')
    plt.title("Unidades vendidas en el tiempo")
    plt.xlabel("Fecha")
    plt.ylabel("Units_Sold")
    plt.subplot(1, 2, 2)
    df.groupby('Date')['Ingresos'].sum().plot(kind='line')
    plt.title("Ingresos Totales")
    plt.xlabel("Fecha")
    plt.ylabel("Ingresos")
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'time_series.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Series de tiempo guardadas en {plot_path}")

def plot_weekday_analysis(df):
    """
    Genera y guarda un gráfico de barras que muestra el promedio de 'Units_Sold' 
    por día de la semana, con etiquetas horizontales y tamaño de letra ajustado.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    import os
    ventas_dow = df.groupby('Día_de_la_semana')['Units_Sold'].mean().sort_values(ascending=False)
    print("Promedio de Units_Sold por día de la semana:")
    print(ventas_dow)
    plt.figure(figsize=(8, 4))
    ax = ventas_dow.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title("Promedio de Units_Sold por Día de la Semana", fontsize=12)
    plt.xlabel("Día de la Semana", fontsize=10)
    plt.ylabel("Units_Sold (Promedio)", fontsize=10)
    plt.xticks(rotation=0, fontsize=10) 
    PLOTS_DIR = 'plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(PLOTS_DIR, 'weekday_analysis.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Análisis por día de la semana guardado en {plot_path}")


def plot_weekday_category_comparison(df):
    """
    Genera y guarda un gráfico de barras agrupadas que muestra el promedio de 
    'Units_Sold' por día de la semana, desglosado por 'Category', 
    con barras lado a lado para cada categoría.

    Args:
        df (pd.DataFrame): DataFrame con el dataset.
    """
    pivot_data = df.pivot_table(index="Día_de_la_semana", 
                                columns="Category", 
                                values="Units_Sold", 
                                aggfunc="mean")
    
    dias_orden = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_data = pivot_data.reindex(dias_orden)
    plt.figure(figsize=(10, 6))
    pivot_data.plot(kind="bar", ax=plt.gca())

    plt.title("Promedio de Units_Sold por Día de la Semana y Categoría", fontsize=14)
    plt.xlabel("Día de la Semana", fontsize=12)
    plt.ylabel("Units_Sold (Promedio)", fontsize=12)
    plt.xticks(rotation=0, fontsize=10)
    plt.legend(title="Categoría", fontsize=10)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, 'weekday_category_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de barras agrupadas guardado en {plot_path}")


def plot_weekday_vs_weekend(df):
    """
    Genera y guarda un gráfico de barras que compara el promedio de ventas (Units_Sold)
    entre días laborables y fines de semana.

    Args:
        df (pd.DataFrame): 
            DataFrame que contiene al menos las columnas 'Día_de_la_semana' y 'Units_Sold'.
    """
    df['Es_fin_de_semana'] = df['Día_de_la_semana'].isin(['Saturday', 'Sunday'])
    weekend_vs_weekday = df.groupby('Es_fin_de_semana')['Units_Sold'].mean()
    print("\nPromedio de ventas: Fines de Semana vs. Días Laborables")
    print(weekend_vs_weekday)

    plt.figure(figsize=(6, 4))
    weekend_vs_weekday.plot(kind='bar', color=['green','blue'], edgecolor='black')
    plt.title("Unidades Vendidas: Fines de Semana vs. Días Laborables")
    plt.xticks([0,1], ['Laborables','Fin de Semana'], rotation=0)
    plt.ylabel("Units_Sold (Promedio)")
    plot_path = os.path.join(PLOTS_DIR, 'weekday_vs_weekend.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de comparación Fines de Semana vs. Días Laborables guardado en {plot_path}")

def plot_store_units_income(df):
    """Genera y guarda una comparación de Unidades Vendidas ('Units_Sold') e Ingresos totales
    ('Ingresos') por tienda en un conjunto de subplots.
    
    Args:
        df (pd.DataFrame):
            DataFrame que contiene al menos las columnas 'Store', 'Units_Sold' y 'Unit_Price'. 
            Si la columna 'Ingresos' no existe, se calculará como 'Units_Sold' * 'Unit_Price'.

    """
    if 'Ingresos' not in df.columns:
        df['Ingresos'] = df['Units_Sold'] * df['Unit_Price']

    pivot_ingresos_unidades = df.groupby('Store')[['Units_Sold','Ingresos']].sum()
    print("\nComparación Unidades Vendidas vs. Ingresos por Tienda:")
    print(pivot_ingresos_unidades)

    pivot_ingresos_unidades.plot(kind='bar', subplots=True, layout=(1,2), figsize=(12,4), legend=False)
    plt.suptitle("Comparación Unidades Vendidas vs. Ingresos por Tienda")
    plot_path = os.path.join(PLOTS_DIR, 'units_vs_income_store.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de Unidades vs. Ingresos por Tienda guardado en {plot_path}")

def main():
    filepath = 'data/data_raw/sales_data.csv'
    df = load_data(filepath)
    
    print("\n--- EDA: Estadísticas Descriptivas ---")
    descriptive_statistics(df)
    
    print("\n--- EDA: Revisión de Valores Nulos ---")
    check_missing_values(df)
    
    print("\n--- EDA: Análisis de Frecuencias ---")
    frequency_analysis(df)
    
    print("\n--- EDA: Boxplots ---")
    plot_boxplots(df)
    
    print("\n--- EDA: Histogramas ---")
    plot_histograms(df)
    
    print("\n--- EDA: Análisis de Ingresos ---")
    plot_income_analysis(df)
    
    print("\n--- EDA: Series de Tiempo ---")
    plot_time_series(df)

    print("\n--- EDA: Análisis de Correlación ---")
    plot_correlation_analysis(df)
    
    print("\n--- EDA: Análisis por Día de la Semana ---")
    plot_weekday_analysis(df)

    print("\n--- EDA: Análisis por Día de la Semana y categoría ---")
    plot_weekday_category_comparison(df)

    print("\n--- EDA: Comparación Fines de Semana vs. Días Laborables ---")
    plot_weekday_vs_weekend(df)
    
    print("\n--- EDA: Comparación Unidades vs. Ingresos por Tienda ---")
    plot_store_units_income(df)

if __name__ == '__main__':
    main()