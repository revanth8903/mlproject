
import sys
import os
from dataclasses import dataclass

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline  ## To Implement the Pipeline, We will be importing the Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts',"preprocessor.pkl")

 ## If we want to create any models, If I want to save that into a pickle file, for that I require any One kind of path  
 ## We have created a "preprocessor.pkl" file , We can also create "model.pkl" file


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):      ## This function which will be responsible for creating  pickle files[which will be responsible in converting categorical features into numerical features]

        '''    
        This Function is responsible for data transformation
        '''
        try:
            numerical_columns =["writing_score","reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

## Created a pipeline which is doing two things, One is handling missing values, Other is doing Standard Scaling and this pipeline needs
## to run on the training data set. fit_transform on the training data set. and just transform on the test data set  

            num_pipeline = Pipeline(
                steps =[
                ("imputer",SimpleImputer(strategy="median")),    ## Imputer will be responsible in handling my missing values 
                ("scaler", StandardScaler())                    ## Here We will be initializing our sample imputer with statergy ="median" 
                                                                ## because there are some outliers
                ]
            )

            logging.info("Numerical columns standard scaling completed")

            cat_pipeline = Pipeline(
                steps =[
                    ("imputer",SimpleImputer(strategy="most_frequent")),    ## Most Frequent is basically the mode
                    ("one_hot_encoder",OneHotEncoder()),                       ## OHE because there are very less number of categories                
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical Columns encoding completed")

            ## To Combine Categorical Pipeline with Numerical Pipeline, We use "Column Transformer" 

            preprocessor = ColumnTransformer(
                [
                    ("num_pipelines", num_pipeline, numerical_columns),
                    ("cat_pipelines",cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining Preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_column_test_df =test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_column_test_df)
            ] 

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)


    

