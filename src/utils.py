import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

import dill

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(object, file)
        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        logging.error(f"Error occurred while saving object: {e}")
        raise CustomException(e, sys) from e 
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        model_report= {}
        for i in range (len(models)):
            model= list(models.values())[i]
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[list(models.keys())[i]]=test_model_score
            
    except Exception as e:
        logging.error(f"Error in {model}: {e}")
        model_report[model] = None
    return model_report