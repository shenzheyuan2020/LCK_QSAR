from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import pickle

def load_processed_data(fp_path='ap_fingerprints.pkl', label_path='labels.pkl'):
    """Load fingerprints and labels."""
    try:
        with open(fp_path, 'rb') as f:
            fingerprints = pickle.load(f)
        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
        print("Data loaded successfully.")
        return fingerprints, labels
    except FileNotFoundError:
        print("Processed data not found. Please check the file path.")
        return None, None

def train_model(fingerprints, labels, test_size=0.1, random_state=None):
    """Train a stacking model and validate its performance."""
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(fingerprints, labels, test_size=test_size, random_state=random_state)
    
    # Define the base learners and stacking model
    base_learners = [
        ('rf', RandomForestRegressor()),
        ('xgb', XGBRegressor(tree_method='gpu_hist', subsample=1, n_estimators=500,
                             min_child_weight=5, max_depth=10, learning_rate=0.01,
                             reg_lambda=0.1, colsample_bytree=0.5, reg_alpha=0)),
        ('knn', KNeighborsRegressor(metric='manhattan', n_neighbors=5, weights='distance')),
        ('lgbm', LGBMRegressor())
    ]
    meta_learner = LinearRegression()
    stacking_model = StackingRegressor(estimators=base_learners, final_estimator=meta_learner)
    
    # Fit the stacking model on the training data
    stacking_model.fit(X_train, y_train)
    
    # Predictions
    train_predictions = stacking_model.predict(X_train)
    test_predictions = stacking_model.predict(X_test)
    
    # Evaluate and print R^2 score and RMSE
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    print("Train R^2 score: ", train_r2)
    print("Test R^2 score: ", test_r2)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")
    
    return stacking_model  # Optionally return the model for further use or saving

# Example usage:
fingerprints, labels = load_processed_data()

if fingerprints is not None and labels is not None:  # Only proceed if data is loaded successfully
    np.random.seed(30)  # Set random seed for reproducibility
    model = train_model(fingerprints, labels, test_size=0.1, random_state=10)
