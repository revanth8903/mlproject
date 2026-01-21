import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


## In Data Ingestion, there should be some inputs that may be probably required by this Data Ingestion Component
## Any Input that I require I will probably give through this particular Data Ingestion Config.

## If We want to use this "Data Ingestion Config", I will probably use a Decorator which is called as "Data Class"
## (Inside a class, To define a class variable, we use __init__ )

## If we use this Dataclass, We will be able to directly define your Class Variable
## train_data_path: (This is one of our class Variable which is defined like this)

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

## Data Ingestion config is just like providing all the input things that is required for the Data Ingestion Component

## Now Data Ingestion Component knows where to save the train,test and data path because of this file path

## If you are only Defining Variables, Then you probably use Data Class, If you have some other functions inside the 
## class, We should go ahead with the "__init__" (constructor) part itself

## Ingestion Config will consist of these three values, 
## When we call these "DataIngestion" class, the three (train_data_path, test_data_path, raw_data_path ) will get initalized.


class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the Data Ingestion Method or Component")
        try:
            df= pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is Completed")
            
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj=DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))



   



        









