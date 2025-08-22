import os
import sys
from dataclasses import dataclass

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
                'XGBRegressor': XGBRegressor(eval_metric='rmse')
            }

            params = {
                'RandomForestRegressor': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                },

                'GradientBoostingRegressor': {
                'n_estimators': [100, 200, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.5],
                'max_depth': [3, 5, 7, 10],
                'subsample': [0.8, 1.0],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
                },

                'AdaBoostRegressor': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1.0],
                'loss': ['linear', 'square', 'exponential']
                },

                'LinearRegression': {
                'fit_intercept': [True, False]
                },

                'Lasso': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                'max_iter': [1000, 5000, 10000],
                },

                'Ridge': {
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0],
                },

                'SVR': {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.2, 0.5]
                },

                'DecisionTreeRegressor': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                },

                'KNeighborsRegressor': {
                'n_neighbors': [3, 5, 7, 9],
                'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
                },

                'XGBRegressor': {
                'n_estimators': [100, 200, 500, 1000],
                'learning_rate': [0.01, 0.05, 0.1, 0.3],
                'max_depth': [3, 5, 7, 10],
                }
            }
    

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                models=models, params=params
                )
            print("DEBUG: model_report =", model_report)

            if not model_report:
                raise CustomException("evaluate_models returned empty report", sys)

            logging.info(f"Model report: {model_report}")

            # Select best model by test R²
            best_model_name = max(model_report, key=lambda name: model_report[name]["test_r2"])
            best_model_info = model_report[best_model_name]
            best_model = best_model_info["model"]
            best_model_score = best_model_info["test_r2"]

            if best_model_score < 0.6:
                logging.warning("Best model has low accuracy. Consider improving feature engineering.")


            logging.info(f"Best model found: {best_model_name} with Test R2 score: {best_model_score}")
            logging.info(f"Best params: {best_model_info['best_params']}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info(f"Best model saved to {self.model_trainer_config.trained_model_file_path}")
            
            save_object(os.path.join('artifacts', 'model_report.pkl'), model_report)
            logging.info("Model report saved to artifacts folder")

            predictions = best_model.predict(X_test)
            r2_square = r2_score(y_test, predictions)

            return {
                "best_model_name": best_model_name,
                "train_r2": best_model_info["train_r2"],
                "test_r2": best_model_score,
                "final_r2": r2_square,
                "best_params": best_model_info["best_params"]
            }
        
        except Exception as e:
            raise CustomException(e, sys)
