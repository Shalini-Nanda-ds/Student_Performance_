import os  # Added import for os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Correct paths for loading the model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            print("Before Loading")
            
            # Load the model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            
            # Transform features and make predictions
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        # Initializing class attributes
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Creating a dictionary with input data
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Converting to a pandas DataFrame
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)
