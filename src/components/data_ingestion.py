#main aim is to read the dataset from specific source
import os 
import sys
from src.exception import customexception
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass #decorator
#inside a class  to define a class variable u write init btt if u use dataclass u will be able to directly define class variable
class DataIngestionConfig: #input part h ki khan pr hm save krege train data ya test data
    train_data_path: str=os.path.join('artifacts',"train.csv")#all the output will be saved in artifact folder
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

#agr khali variable ho toh ap dataclass use kr lo aur bhi functionallity use krni h toh init wla use kro

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()#defining variable

    def initiate_data_ingestion(self): #agr apka data is stored in some databases so we will write this code to read that data
            logging.info("entered the data ingestion method or component")
            try:
                df=pd.read_csv('notebook/data/stud.csv')
                logging.info('read the dataset as dataframe')

                os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)#agar already hoga toh hm vhi rkhege use delete krke vapis se create ni krege
                df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

                logging.info("train test split initiated")
                train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

                train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                logging.info("ingestion of data is completed")

                return(
                    self.ingestion_config.train_data_path,
                    self.ingestion_config.test_data_path
                )


            except Exception as e:
                raise customexception(e,sys)
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_tranformation=DataTransformation()
    train_arr,test_arr,_=data_tranformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

