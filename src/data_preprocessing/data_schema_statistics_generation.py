import pandas as pd
import pandera as pa
from pandera.typing import DataFrame
import sys
import os
from logger import logging
from exceptions import CustomException

def generate_schema(df):
#Generate a Pandera schema based on the provided DataFrame
    try:
        logging.info("Generating data schema")
        
        # Create a schema class dynamically based on the DataFrame
        schema_dict = {}
        
        # Define checks for each column
        if 'age' in df.columns:
            schema_dict['age'] = pa.Column(int, pa.Check.in_range(0, 100), coerce=True)
            
        if 'gender' in df.columns:
            schema_dict['gender'] = pa.Column(str, pa.Check.isin(["Male", "Female"]), coerce=True)
            
        if 'race' in df.columns:
            schema_dict['race'] = pa.Column(str, pa.Check.isin(["Caucasian", "AfricanAmerican", "Other"]), coerce=True)
            
        if 'admission_type_id' in df.columns:
            schema_dict['admission_type_id'] = pa.Column(
                str, 
                pa.Check.isin(["Emergency", "Urgent", "Elective", "New Born", "Not Available", "Trauma Center"]),
                coerce=True
            )
            
        if 'discharge_disposition_id' in df.columns:
            schema_dict['discharge_disposition_id'] = pa.Column(
                str, 
                pa.Check.isin(["Discharged to Home", "Care/Nursing", "Other", "Unknown"]),
                coerce=True
            )
            
        if 'admission_source_id' in df.columns:
            schema_dict['admission_source_id'] = pa.Column(
                str, 
                pa.Check.isin(["Referral", "Emergency room", "Others"]),
                coerce=True
            )
            
        if 'diag_1' in df.columns:
            schema_dict['diag_1'] = pa.Column(
                str, 
                pa.Check.isin(["Circulatory", "Respiratory", "Diabetes", "Injury", "Musculoskelatal", "Genitourinary", "Neoplasms", "Others"]),
                coerce=True
            )
            
        if 'readmitted' in df.columns:
            schema_dict['readmitted'] = pa.Column(str, pa.Check.isin(["Yes", "No"]), coerce=True)
        
        # Create the schema
        schema = pa.DataFrameSchema(schema_dict)
        
        logging.info("Data schema generated successfully")
        return schema
    
    except Exception as e:
        logging.error(f"Error in generate_schema: {str(e)}")
        raise CustomException(e, sys)

def prepare_data_splits(df):
#Split the data into training, evaluation, and serving sets
    try:
        logging.info("Preparing data splits")

        total_len = len(df)
        train_len = int(total_len * 0.6)
        eval_len = int(total_len * 0.2)
        
        train_df = df.iloc[:train_len].reset_index(drop=True)
        eval_df = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)
        
        # For serving data, remove the target column if it exists
        serving_df = df.iloc[train_len + eval_len:].reset_index(drop=True)
        if 'readmitted' in serving_df.columns:
            serving_df = serving_df.drop(columns=['readmitted'])
        
        logging.info(f"Prepared training data with shape {train_df.shape}")
        logging.info(f"Prepared evaluation data with shape {eval_df.shape}")
        logging.info(f"Prepared serving data with shape {serving_df.shape}")
        
        return train_df, eval_df, serving_df
    
    except Exception as e:
        logging.error(f"Error in prepare_data_splits: {str(e)}")
        raise CustomException(e, sys)

def validate_data_schema(df):
#Validate data schema across training, evaluation, and serving splits
    try:
        logging.info("Beginning data schema validation process")
        
        # Prepare data splits
        train_df, eval_df, serving_df = prepare_data_splits(df)
        
        # Generate schema from training data
        schema = generate_schema(train_df)
        logging.info("Generated schema from training data")
        
        # Validate training data
        try:
            validated_train_df = schema.validate(train_df)
            logging.info("Training data validation successful")
        except pa.errors.SchemaError as e:
            logging.warning(f"Training data validation issues: {str(e)}")
        
        # Validate evaluation data
        try:
            validated_eval_df = schema.validate(eval_df)
            logging.info("Evaluation data validation successful")
        except pa.errors.SchemaError as e:
            logging.warning(f"Evaluation data validation issues: {str(e)}")
            
        # Validate serving data (schema without target column)
        if 'readmitted' in schema.columns:
            serving_schema = schema.remove_columns(['readmitted'])
        else:
            serving_schema = schema
            
        try:
            validated_serving_df = serving_schema.validate(serving_df)
            logging.info("Serving data validation successful")
        except pa.errors.SchemaError as e:
            logging.warning(f"Serving data validation issues: {str(e)}")
        
        # Generate statistics about the data
        logging.info(f"Processed data shape: {train_df.shape}")
        if 'readmitted' in train_df.columns:
            readmission_rate = train_df['readmitted'].value_counts(normalize=True).to_dict()
            logging.info(f"Readmission rate: {readmission_rate}")
        
        # Combine all data back for return
        processed_df = pd.concat([train_df, eval_df, serving_df], ignore_index=True)
        
        logging.info("Data schema validation completed successfully")
        return processed_df
    
    except Exception as e:
        logging.error(f"Error in validate_data_schema: {str(e)}")
        raise CustomException(e, sys)
