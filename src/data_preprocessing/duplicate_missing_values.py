import pandas as pd
import numpy as np
from logger import logging
from exceptions import CustomException
import sys

def missingVal(df):
    try:
        
        #Case 1: Missing Values
        #Dropping columns which has more N/A values
        if 'max_glu_serum' in df.columns:
            del df['max_glu_serum'] ## 96420 Droping N/A
        if 'A1Cresult' in df.columns:
            del df['A1Cresult']  ## 84748 Droping N/A
        logging.info("Dropped N/A Values")
        
        #Case 2: Inconsistent Values
        ## Droping columns which has more '?' values 
        if 'weight' in df.columns:
            del df['weight']  ## #98569 '?' values
        if 'payer_code' in df.columns:
            del df['payer_code'] 
        if 'medical_specialty' in df.columns:
            del df['medical_specialty'] 
        if 'examide' in df.columns:
            del df['examide'] 
        if 'citoglipton' in df.columns:
            del df['citoglipton']
        if 'glimepiride-pioglitazone' in df.columns:
            del df['glimepiride-pioglitazone']
        logging.info("Dropped inconsistent Values")

        #CASE 3: Setting Target Value
        logging.info("Setting up Target value")
        df["readmitted"] = np.where(
        (df["readmitted"] == ">30") | (df["readmitted"] == "<30"),
                            "Yes",
                            "No"
        )
        

    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

    return df

def duplicates(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info("File reading begins")

        duplicates=df.duplicated()
        count= duplicates.shape[0]
        if count<0:
            logging.warning(f"Found {count} duplicate rows!")
        else:
            logging.info("No duplicate data found")
        
        # drop
        drop_col =['encounter_id', 'patient_nbr']
        df.drop(drop_col, axis=1, inplace=True)
        #logging.info(f"Dropped columns {drop_col}")
    
        unwanted =[11,19,20,21]# ids which are expired, homefacility, hospice etc
        df=df[~df['discharge_disposition_id'].isin(unwanted)]
        df.reset_index(drop=True, inplace=True)
        #logging.info("desposition ids")
        return df
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)
    
   
    
