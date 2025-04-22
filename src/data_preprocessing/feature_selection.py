import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency,ttest_ind
import os
from logger import logging
from exceptions import CustomException
import sys


def feature_selection(df):
    try:
        logging.info("Starting feature selection process...")
        df['readmitted']=df['readmitted'].apply(lambda x:'No' if x==0
                       else 'Yes')
    
        num_cols = df.select_dtypes(include=['number']).columns
        cat_cols = df.select_dtypes(include=['object']).columns
    
        selected_cols=[]
        for i in cat_cols:
            crosstab_values=pd.crosstab(df[i],df['readmitted']).values
            chi2,p_value,dof,expected=chi2_contingency(observed=crosstab_values,correction=False)
    
            if p_value<0.05:
                selected_cols.append(i) 
    
        cat_size=len(selected_cols)
        if cat_size==0:
            logging.info("No Categorical columns was selected")
        else: 
            logging.info("Significant categorical features added to the list.")
    
        for i in num_cols:
            gr1 = df[df['readmitted'] == 'No'][i]
            gr2 = df[df['readmitted'] == 'Yes'][i]

            t_stat, pval = ttest_ind(gr1, gr2, equal_var=False) 
        
            if pval < 0.05:
                selected_cols.append(i)
    
        if len(selected_cols)==cat_size:
            logging.info("No Numerical columns was selected")
        else: 
            logging.info("Significant numerical features added to the list.")
    
        df_fs=df[selected_cols]
    
        logging.info("Feature selection completed successfully.")

        return df_fs

    except Exception as e:
        logging.error("An error occurred during feature selection.")
        raise CustomException(e, sys)