def get_grid_params():
    """Define los grids de hiperparámetros para cada modelo.

    Returns:
        dict: Diccionario con grids de hiperparámetros.
    """
    param_grid = {
        'Linear Regression': {},
        'Lasso': {
            'regressor__alpha': [0.01, 0.1, 1, 10]
        },
        'Ridge': {
            'regressor__alpha': [0.01, 0.1, 1, 10]
        },
        'Decision Tree': {
            'regressor__max_depth': [None, 5, 10, 20],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'Random Forest': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 5, 10, 20],
            'regressor__min_samples_split': [2, 5, 10]
        },
        'Gradient Boosting': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 0.2],
            'regressor__max_depth': [3, 5, 7]
        },
        'AdaBoost': {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__learning_rate': [0.01, 0.1, 1]
        }
    }
    return param_grid