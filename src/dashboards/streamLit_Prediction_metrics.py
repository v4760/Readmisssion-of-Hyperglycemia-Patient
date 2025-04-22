import os
import io
import joblib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sklearn.preprocessing import RobustScaler
from google.cloud import storage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from exceptions import CustomException
from logger import logging
# Set logging
#logging.basicConfig(level=logging.INFO)

# -------------------------------------
# Streamlit Cache Functions
# -------------------------------------

@st.cache_resource
def get_gcs_client():
    KEY_PATH = "config/key.json"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
    return storage.Client()

@st.cache_resource
def get_engine():
    load_dotenv()
    DB_NAME = os.getenv('DB_NAME')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    engine = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    return engine

# -------------------------------------
# GCS: Load Scaler
# -------------------------------------

@st.cache_resource
def load_scaler_from_gcs(bucket_name: str, file_path: str):
    try:
        client = get_gcs_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        contents = blob.download_as_bytes()
        scaler = joblib.load(io.BytesIO(contents))
        return scaler
    except Exception as e:
        logging.error("Failed to load scaler parameters from GCS.")
        raise e

# -------------------------------------
# DB: Load Data
# -------------------------------------

@st.cache_data
def load_data_into_df():
    try:
        engine = get_engine()
        query = """
        SELECT * FROM patients_data
        """
        df_scaled = pd.read_sql(query, engine)
        logging.info(f"Fetched {len(df_scaled)} rows from the database.")

        return df_scaled
    except Exception as e:
        logging.error("Failed to fetch data from DB.")
        raise e

# -------------------------------------
# Reverse Scaler
# -------------------------------------

@st.cache_data
def reverse_scaler():
    try:
        df_scaled = load_data_into_df()
        scaler = load_scaler_from_gcs('readmission_prediction', 'models/best_xgboost_model/scalar_weight.pkl')
        features = scaler.feature_names_in_
        df_original = df_scaled.copy()
        df_original[features] = scaler.inverse_transform(df_scaled[features])

        return df_original
    except Exception as e:
        logging.error("Error in reverse scaling.")
        raise e

# -------------------------------------
# Streamlit Dashboard
# -------------------------------------
def set_page_style():
    st.markdown("""
    <style>
        /* Global font override for most text */
        html, body, [class*="css"]  {
            font-family: 'Times New Roman', serif;
        }

        /* Titles */
        .stApp h1 {
            font-size: 36px !important;
            font-weight: bold;
        }

        /* Subheaders */
        .stApp h2 {
            font-size: 24px !important;
            font-weight: bold;
        }

        /* Metric components */
        .stMetric {
            font-size: 18px !important;
            font-weight: bold;
        }

        /* Button styling */
        .stButton > button {
            font-size: 16px;
            font-family: 'Times New Roman', serif;
        }
    </style>
    """, unsafe_allow_html=True)


# Your dashboard creation function


def dashboard_creation():
    try:
        # Set the custom styles for fonts
        set_page_style()
        
        st.title("📊 Patient Readmission Dashboard")

        df = reverse_scaler()  # Assuming reverse_scaler() is a function that loads the data
        df = df.dropna()
        df = df[ 
            (df['f_name'].fillna('') != '') & 
            (df['l_name'].fillna('') != '') & 
            (df['dob'].astype(str).fillna('') != '') & 
            (df['readmitted'].astype(str).fillna('') != '')
        ]

        # Monitoring Patient Visits
        df['patient_key'] = df['f_name'] + "_" + df['l_name'] + "_" + df['dob'].astype(str)

        selected_patient = st.selectbox("Select Patient", df['patient_key'].unique())

        # Patient history
        patient_history = df[df['patient_key'] == selected_patient]
        patient_history_sorted = patient_history.sort_values('time_in_hospital')

        patient_history['readmitted'] = patient_history['readmitted'].astype(int)
        patient_history['health_index'] = patient_history['health_index'].apply(lambda x: f"{x:.2f}")

        st.subheader("📄 Patient Visit History")
        styled_df = patient_history[[
            'f_name','l_name', 'dob', 'health_index'
        ]].style \
            .background_gradient(cmap='coolwarm', subset=['health_index']) \
            .set_properties(**{'font-family': 'Arial', 'font-size': '14px'})

        # Display the styled dataframe
        st.dataframe(styled_df, use_container_width=True)
        # Normalize the medication count values to 0–1 range

        # Creating columns for layout
        col1, col2 = st.columns([2, 2])  # Adjust width ratio to be more balanced

        with col1:
            st.subheader("📉 Health Index vs. Readmission Status")

            # Create the figure and axis for scatter plot
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust the size as needed

            # Color variation based on readmission status
            colors = patient_history_sorted['readmitted'].map({0: 'green', 1: 'red'})

            # Scatter plot for Readmission vs Health Index
            scatter = ax.scatter(patient_history_sorted['readmitted'], patient_history_sorted['health_index'], 
                                color=colors, alpha=0.7, edgecolors='w', s=100)  # Swapped X and Y axes

            # Set title and labels
            ax.set_xlabel('Readmission Status (0 or 1)')
            ax.set_ylabel('Health Index')

            # Set x-axis to display only 0 and 1
            ax.set_xticks([0, 1])

            # Optionally, add a legend for clarity
            red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Readmitted (1)')
            green_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Not Readmitted (0)')
            ax.legend(handles=[red_patch, green_patch], loc='upper right')

            # Display the plot in Streamlit
            st.pyplot(fig)  # Pass the figure explicitly

        with col2:
                # Preprocessing - Prediction Accuracy
            # Readmission Analysis
            st.subheader("📉 Readmission Analysis")

            # Calculate the value counts for readmitted (0 and 1)
            readmitted_percentage = patient_history['readmitted'].value_counts(normalize=True)  # Normalize to get percentages

            # Force both 0 and 1 to appear, even if one is missing
            readmitted_percentage = readmitted_percentage.reindex([0, 1], fill_value=0)

            # Convert to percentages
            readmitted_percentage = readmitted_percentage * 100

            # Create a Matplotlib figure with custom size for better visual balance
            fig, ax = plt.subplots(figsize=(5, 4))  # Adjust the size to make the chart smaller

            # Plot the histogram with color variation
            readmitted_percentage.plot(kind='bar', ax=ax, color=['green', 'red'], alpha=0.7)

            # Show the plot in Streamlit
            st.pyplot(fig)

        col3, col4 = st.columns([2, 2])  # Define two columns for side-by-side layout

        # Column 3: Avg Diagnoses by Severity Level
        with col3:
            st.subheader("🩺 Diagnoses by Severity Level")

            # Plotting severity distribution using matplotlib
            fig, ax = plt.subplots(figsize=(5, 4))
            avg_diag = patient_history.groupby('severity_of_disease')['number_diagnoses'].mean().sort_index()
            avg_diag.plot(kind='bar', color='salmon', ax=ax)

            ax.set_title('Avg Diagnoses by Severity Level')
            ax.set_xlabel('Severity Level')
            ax.set_ylabel('Avg No. of Diagnoses')
            st.pyplot(fig)

        # Column 4: Disease Severity vs Health Index Scatter Plot (with red/green color)
        with col4:
            st.subheader("🚑 Disease Severity vs Health Index")
         
            # Generate a list of RGBA colors based on normalized values
            # Assign colors based on severity_of_disease
            patient_history['color'] = patient_history['severity_of_disease'].apply(lambda x: 'salmon' if x == 1 else 'green')

            # Create a scatter chart with health_index vs severity_of_disease and use color mapping
            st.scatter_chart(patient_history[['health_index', 'severity_of_disease']].assign(color=patient_history['color']))

        col5, col6= st.columns([2, 2])

        with col5:
            st.subheader("💊 Medication Usage Overview")
            medication_counts = patient_history[['metformin', 'repaglinide', 'glipizide', 'glyburide', 
                                        'pioglitazone', 'rosiglitazone', 'acarbose', 'insulin']].sum()
            medication_counts = medication_counts.abs()
            st.write("Medication Counts (ensured positive values):")
            st.write(medication_counts)

        with col6:
            st.subheader("📊 Medication Usage Distribution")
    
            fig, ax = plt.subplots(figsize=(5, 4))
            medication_counts.plot(kind='bar', color='salmon', ax=ax)
            ax.set_xlabel('Medications')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        
        # Model Performance Metrics tab
        precision = precision_score(df['readmitted'], df['predict'])
        recall = recall_score(df['readmitted'], df['predict'])
        f1 = f1_score(df['readmitted'], df['predict']) 
        accuracy = (df['predict'] == df['readmitted']).mean()

        st.subheader(f"🔍Overall Model Accuracy: {accuracy * 100:.0f}%")

        with st.expander("🔍 View Model Performance Metrics"):
            st.metric(label="🎯 Precision", value=f"{precision:.2f}")
            st.metric(label="🔁 Recall", value=f"{recall:.2f}")
            st.metric(label="📊 F1 Score", value=f"{f1:.2f}")
        



    except Exception as e:
        logging.error("Dashboard creation failed.", exc_info=True)
        st.error("An error occurred while generating the dashboard.")

# Call the function
dashboard_creation()



