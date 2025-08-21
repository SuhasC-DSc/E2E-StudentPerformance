import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], train_array[:, -1],
                test_array[:, :-1], test_array[:, -1]
            )

            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'Lasso': Lasso(),
                'Ridge': Ridge(),
                'SVR': SVR(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=0),
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models
            )
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model_name = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model_name
            ) 

            predictions = best_model_name.predict(X_test)
            r2_square = r2_score(y_test, predictions)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys) from e