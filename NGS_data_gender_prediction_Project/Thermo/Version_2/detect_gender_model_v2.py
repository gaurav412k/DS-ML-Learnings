import pandas as pd
import os
import numpy as np
import pickle
import sys
import joblib 


import joblib
import pandas as pd
import os

def load_and_process_single_tsv(tsv_file) -> pd.DataFrame:
    """
    Load and process a single TSV file.

    Parameters:
    tsv_file (str): The path to the single TSV file.

    Returns:
    pd.DataFrame: A DataFrame containing the processed data.
    """
    # Define the columns of interest
    columns_of_interest = ['chromosome', 'readcount', 'start']

    try:
        # Assume `tsv_file` is already a DataFrame (replace with `pd.read_csv()` if needed)
        dtf = tsv_file

        # Keep only the columns of interest if they exist in the file
        dtf = dtf[[col for col in columns_of_interest if col in dtf.columns]]

        # Filter for rows where 'chromosome' is either 'chrX' or 'chrY'
        dtf = dtf[dtf['chromosome'].isin(['chrX', 'chrY'])]

        # Safely assign the 'sample' column
        dtf.loc[:, 'sample'] = "Sample"

        # Group by chromosome and sample and aggregate
        reduced_row = dtf.groupby(['chromosome', 'sample']).agg({
            'readcount': 'sum',  # Sum readcount values
        }).reset_index()

        # Pivot the DataFrame to get chromosomes as columns
        pivoted_df = reduced_row.pivot_table(index=['sample'], columns='chromosome', values='readcount')
        pivoted_df.reset_index(inplace=True)

        # Filter for 'chrY' with specific start values for the SRY gene
        sry_df = dtf[(dtf['chromosome'] == 'chrY') & (dtf['start'].isin([2000001]))][['readcount', 'sample']]
        sry_df.rename(columns={'readcount': 'sry_count'}, inplace=True)
        sry_df.reset_index(drop=True, inplace=True)

        # Merge the pivoted DataFrame with the SRY data
        final_df = pd.merge(pivoted_df, sry_df, on='sample', how='inner')

        # Add fractional columns
        final_df["frac_X"] = final_df["chrX"] / (final_df["chrX"] + final_df["chrY"])
        final_df["frac_Y"] = final_df["chrY"] / (final_df["chrX"] + final_df["chrY"])

        return final_df

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def predict_gender(input_df: pd.DataFrame, combined_model_path: str) -> str:
    try:
        input_df = load_and_process_single_tsv(input_df)

        # Load model and scaler
        loaded_data = joblib.load(combined_model_path)
        loaded_model = loaded_data['svm_model']
        loaded_scaler = loaded_data['scaler']

        # Ensure required columns
        required_columns = ['chrX', 'chrY', 'sry_count', 'frac_X', 'frac_Y']
        if not all(col in input_df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")

        # Standardize features
        scaled_features = loaded_scaler.transform(input_df[required_columns])

        # Predict gender
        predictions = loaded_model.predict(scaled_features)
        input_df['predicted_gender'] = predictions

        # Check prediction results
        gender = input_df['predicted_gender'].astype(int)
        male = gender[gender == 1]
        female = gender[gender == 0]

        if not male.empty:
            return "Male"
        if not female.empty:
            return "Female"

        return "Unknown"

    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error"

if __name__ == "__main__":
    test_df = pd.read_csv(sys.argv[1], sep="\t")
    print(predict_gender(test_df,"svm_model_thermo_v2"))

