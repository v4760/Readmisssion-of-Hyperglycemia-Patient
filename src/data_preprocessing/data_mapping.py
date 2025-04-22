import os
from logger import logging
from exceptions import CustomException
import sys


def clean_gender(df):
    try:
        logging.info('Running clean_gender')
        #Remove Unknown/Invalid gender entries and reset index 
        df = df.drop(df.loc[df["gender"]=="Unknown/Invalid"].index, axis=0)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)


def map_age_ranges(df):
    #Convert age ranges to numerical middle values 
    try:
        logging.info('running map_age_range')
        age_mapping = {
            "[70-80)": 75, "[60-70)": 65, "[50-60)": 55, "[80-90)": 85,
            "[40-50)": 45, "[30-40)": 35, "[90-100)": 95, "[20-30)": 25,
            "[10-20)": 15, "[0-10)": 5
        }
        df['age'] = df['age'].replace(age_mapping)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

def map_race(df):
    try:
        logging.info('running map_race')
        #Simplify race categories 
        race_mapping = {"Asian": "Other", "Hispanic": "Other", '?': 'Other'}
        df['race'] = df['race'].replace(race_mapping)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

def map_admission_type(df):
    try:
        logging.info('running map_admission_type')
        #Map admission type IDs to descriptive categories 
        admission_mapping = {
            1.0: "Emergency", 2.0: "Urgent", 3.0: "Elective",
            4.0: "New Born", 5.0: "Not Available", 6.0: "Not Available",
            7.0: "Trauma Center", 8.0: "Not Available"
        }
        df['admission_type_id'] = df['admission_type_id'].replace(admission_mapping)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

def map_discharge(df):
    try:
        logging.info('running map_discharge')
        #Map discharge disposition IDs to simplified categories
        discharge_mapping = {
            1: "Discharged to Home", 6: "Discharged to Home",
            8: "Discharged to Home", 13: "Discharged to Home",
            18: "Unknown", 25: "Unknown", 26: "Unknown",
            2: "Other", 3: "Care/Nursing", 4: "Care/Nursing",
            5: "Care/Nursing", 7: "Other", 9: "Other",
            10: "Other", 11: "Other", 12: "Other",
            14: "Care/Nursing", 15: "Other", 16: "Other",
            17: "Other", 20: "Other", 21: "Other",
            22: "Care/Nursing", 23: "Care/Nursing", 24: "Care/Nursing",
            27: "Other", 28: "Other", 29: "Other", 30: "Other"
        }
        df["discharge_disposition_id"] = df["discharge_disposition_id"].replace(discharge_mapping)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

def map_admission_source(df):
    try:
        #Map admission source IDs to simplified categories
        logging.info('running map_admission')
        def map_source(x):
            if x in [1, 2, 3]:
                return 'Referral'
            elif x == 7:
                return 'Emergency room'
            else:
                return 'Others'
        
        df['admission_source_id'] = df['admission_source_id'].apply(map_source)
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)
    
def clean_diagnosis(df):
    #Clean and handle missing values in diagnosis columns #
    # Drop inconsistent diagnosis rows
    try:
        logging.info('running clean_diagnosis')
        diag1_index = df[(df['diag_1']=='?') & ((df['diag_3']!='?') | (df['diag_2']!='?'))].index
        df.drop(diag1_index, inplace=True)
        
        diag2_index = df[(df['diag_2']=='?') & (df['diag_3']!='?')].index
        df.drop(diag2_index, inplace=True)
        
        # Fill missing values with mode
        for col in ['diag_1', 'diag_2', 'diag_3']:
            mode_val = df[col].mode()[0]
            df[col] = df[col].replace('?', mode_val)
        
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)
    
def map_diagnosis_categories(x):
    #Map diagnosis codes to disease categories
    try:
        if str(x).startswith('V') or str(x).startswith('E'):
            return 'Others'
        
        x_val = int(float(x))
        
        if (390 <= x_val <= 459) or x_val == 785:
            return 'Circulatory'
        elif (460 <= x_val <= 519) or x_val == 786:
            return 'Respiratory'
        elif x_val == 250:
            return 'Diabetes'
        elif 800 <= x_val <= 999:
            return 'Injury'
        elif 710 <= x_val <= 739:
            return 'Musculoskelatal'
        elif (580 <= x_val <= 629) or x_val == 788:
            return 'Genitourinary'
        elif 140 <= x_val <= 239:
            return 'Neoplasms'
        else:
            return 'Others'
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

def process_diagnosis_columns(df):
    try:
        logging.info('process_diagnosis_columns')

        # Drop rows where any of diag_1, diag_2, diag_3 contain '?'
        df = df[~df[['diag_1', 'diag_2', 'diag_3']].isin(['?']).any(axis=1)]
        
        # Apply diagnosis mapping to all diagnosis columns
        for col in ['diag_1', 'diag_2', 'diag_3']:
            df[col] = df[col].apply(map_diagnosis_categories)

        return df
    except Exception as e:
        logging.error(f'Error processing diagnosis columns: {e}')
        raise CustomException(e, sys)


# Main function to process data mapping
def process_data_mapping(df):
    
    try:
        logging.info("Starting data mapping process")

        processing_functions = [
            clean_gender, map_age_ranges, map_race, map_admission_type,
            map_discharge, map_admission_source, clean_diagnosis, 
            process_diagnosis_columns
        ]
        for func in processing_functions:
            df = func(df)
            logging.info(f"Completed {func.__name__}")
        

        return df 
        
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)
