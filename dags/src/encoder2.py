import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
import sys

def target_encoding(df):
    try:
        df["readmitted"] = df["readmitted"].map({"Yes": 1, "No": 0})
        # Resetting the indiced
        df=df.reset_index(drop=True)
        
        for col in df.loc[:,"metformin":'metformin-pioglitazone'].columns:
            df[col] = df[col].apply(lambda x : 10 if x == 'Up'
                                              else ( -10 if x == 'Down'
                                              else ( 0 if x == 'Steady'
                                              else  -20)))

        
    except Exception as e:
        logging.info('__.__Error occoured__.__')
        raise CustomException(e,sys)

    return df
#TEST FI



        
