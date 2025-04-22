import pandas as pd
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set to DEBUG to capture all logs
                    filename="logsty.log",
                    filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Also log to console for debugging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def duplicate_data():
    try:
        logging.info("File reading begins...")  # This should be logged
        df = pd.read_csv("data/diabetic_data.csv") 
        logging.info("File successfully read!")

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            logging.warning(f"Found {duplicate_count} duplicate rows!")
        else:
            logging.info("No duplicate data found.")

        # Drop unnecessary columns
        drop_col = ['encounter_id', 'patient_nbr']
        df.drop(columns=drop_col, inplace=True)
        logging.info(f"Dropped columns: {drop_col}")

        # Filter out unwanted discharge disposition IDs
        unwanted = [11, 19, 20, 21]  # IDs representing expired, home facility, hospice, etc.
        df = df[~df['discharge_disposition_id'].isin(unwanted)]
        df.reset_index(drop=True, inplace=True)
        logging.info("Filtered unwanted disposition IDs and reset index.")

        # Save cleaned data
        df.to_csv("data/cleaned_diabetic_data.csv", index=False)
        logging.info("File saved successfully as 'cleaned_diabetic_data.csv'")

        logging.shutdown()  # Ensure logs are written
        return 0  # Successful execution

    except FileNotFoundError:
        logging.error("FileNotFoundError: The file does not exist in the folder.", exc_info=True)
        logging.shutdown()  # Ensure error is logged
        return -1  # Indicating failure

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        logging.shutdown()  # Ensure error is logged
        return -1

# Run function for testing
if __name__ == "__main__":
    result = duplicate_data()
    print(f"Script execution result: {result}")  # Debugging print
