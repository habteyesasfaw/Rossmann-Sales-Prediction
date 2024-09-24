from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

def build_pipeline(model):
    return Pipeline([('model', model)])

def train_model(X, y, model, param_grid=None):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model
    val_score = best_model.score(X_val, y_val)
    return best_model, val_score

def train_random_forest(X, y):
    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None]
    }
    return train_model(X, y, model, param_grid)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.show()

def confidence_intervals(predictions, n_bootstrap=1000):
    bootstrapped_means = [np.mean(np.random.choice(predictions, size=len(predictions), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(bootstrapped_means, 2.5)
    upper = np.percentile(bootstrapped_means, 97.5)
    return lower, upper

def save_model(model, folder_path="models/"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_filename = f"{folder_path}/model_{timestamp}.pkl"
    joblib.dump(model, model_filename)
