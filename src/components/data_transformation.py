import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str =os.path.join('artifacts', 'preprocessor.pkl')  

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            logging.info("Numerical and categorical columns identified")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])  
            logging.info("Numerical and categorical pipelines created with imputation and scaling")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numerical_columns),
                    ('cat', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while creating data transformer object: {e}")
            raise CustomException(e, sys) from e
        
    def initiate_data_transformation(self, train_path, test_path):
        logging.info("Data Transformation method started")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and test datasets read as pandas DataFrames")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Input and target features separated for train and test datasets")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Data transformation applied to input features")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Train and test arrays created with transformed features")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return ( 
            train_arr,
            test_arr,
            self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f"Error occurred during data transformation: {e}")
            raise CustomException(e, sys) from e
        
   