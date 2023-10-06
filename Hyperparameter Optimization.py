from sklearn.model_selection import GridSearchCV
import numpy as np

def optimize_hyperparameters(X_train, y_train):
    """Optimize hyperparameters for the base learners using grid search."""
    # Define parameter grid for each base learner
    param_grid = {
        'rf': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        },
        'xgb': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'learning_rate': [0.01, 0.1]
        },
        'knn': {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        },
        'lgbm': {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1]
        }
    }

    optimized_learners = []

    # Define base learners
    base_learners = [
        ('rf', RandomForestRegressor()),
        ('xgb', XGBRegressor(tree_method='gpu_hist')),
        ('knn', KNeighborsRegressor(metric='manhattan')),
        ('lgbm', LGBMRegressor())
    ]
    
    # Optimize hyperparameters for each learner
    for name, model in base_learners:
        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid[name],
                            scoring='neg_mean_squared_error',  # Use MSE as optimization metric
                            n_jobs=-1,  # Use all available CPUs
                            cv=3,  # Use 3-fold cross-validation
                            verbose=1)  # Print optimization process
        grid.fit(X_train, y_train)
        
        best_params = grid.best_params_
        best_model = grid.best_estimator_
        
        print(f"\nBest parameters for {name}: {best_params}\n")
        optimized_learners.append((name, best_model))

    return optimized_learners

# Example usage:
fingerprints, labels = load_processed_data()

if fingerprints is not None and labels is not None:  # Only proceed if data is loaded successfully
    np.random.seed(30)  # Set random seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(fingerprints, labels, test_size=0.1, random_state=10)
    optimized_learners = optimize_hyperparameters(X_train, y_train)
