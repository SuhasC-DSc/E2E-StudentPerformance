import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys) from e 
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            # Check if model has a parameter grid
            if model_name in params and params[model_name]:
                para = params[model_name]
                logging.info(f"Running GridSearchCV for {model_name} with {len(para)} parameter sets")

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=3,
                    scoring='r2',
                    n_jobs=-1,
                    error_score='raise'
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
                best_params = gs.best_params_
                logging.info(f"GridSearchCV completed for {model_name}")
            else:
                # No parameter grid: train with default parameters
                logging.info(f"No param grid for {model_name}, fitting default model")
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}

            # Evaluate
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            model_report[model_name] = {
                "model": best_model,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "best_params": best_params
            }

            logging.info(f"{model_name} - Train R2: {train_r2:.4f}, Test R2: {test_r2:.4f}")

        return model_report

    except Exception as e:
        logging.error(f"Error in evaluate_models: {e}")
        raise CustomException(e, sys) from e