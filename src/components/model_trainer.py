import os 
import sys
from dataclasses import dataclass

from catboost import  CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import customexception
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trainer_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],#all rows and all columns except last one
                train_array[:,-1],#all rows and last column
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Linear Regression": LinearRegression(),
                "gradient boosting":GradientBoostingRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=models)#evaluate model is func which we will create in utils

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise customexception("no best model found")
            
            logging.info(f"best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trainer_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)
            r2=r2_score(y_test,predicted)
            return r2
        except Exception as e:
            raise customexception(e,sys)