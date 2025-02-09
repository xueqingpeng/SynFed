import pandas as pd
import argparse

# Mapping dictionary for values where keys are column indices (numbers)
value_dict = {
    0: {'A11': 'smaller than 0 DM',
        'A12': 'bigger than 0 DM but smaller than 200 DM',
        'A13': 'bigger than 200 DM OR salary assignments for at least 1 year',
        'A14': 'no checking account'},
    2: {'A30': 'no credits taken or all credits paid back duly',
        'A31': 'all credits at this bank paid back duly',
        'A32': 'existing credits paid back duly till now',
        'A33': 'delay in paying off in the past',
        'A34': 'critical account or other credits existing'},
    3: {'A40': 'new car',
        'A41': 'used car',
        'A42': 'furniture or equipment',
        'A43': 'radio or television',
        'A44': 'domestic appliances',
        'A45': 'repairs',
        'A46': 'education',
        'A47': 'vacation',
        'A48': 'retraining',
        'A49': 'business',
        'A410': 'others'},
    5: {'A61': 'smaller than 100 DM',
        'A62': 'bigger than 100 smaller than 500 DM',
        'A63': 'bigger than 500 smaller than 1000 DM',
        'A64': 'bigger than 1000 DM',
        'A65': 'unknown or no savings account'},
    6: {'A71': 'unemployed',
        'A72': 'smaller than 1 year',
        'A73': 'bigger than 1 smaller than 4 years',
        'A74': 'bigger than 4 smaller than 7 years',
        'A75': 'bigger than 7 years'},
    8: {'A91': 'male divorced or separated',
        'A92': 'female divorced or separated or married',
        'A93': 'male and single',
        'A94': 'male and married or widowed',
        'A95': 'female and single'},
    9: {'A101': 'none',
        'A102': 'co-applicant',
        'A103': 'guarantor'},
    11: {'A121': 'real estate',
         'A122': 'building society savings agreement or life insurance',
         'A123': 'car or other',
         'A124': 'unknown or no property'},
    13: {'A141': 'bank',
         'A142': 'stores',
         'A143': 'none'},
    14: {'A151': 'rent',
         'A152': 'own',
         'A153': 'for free'},
    16: {'A171': 'unemployed or unskilled or non-resident',
         'A172': 'unskilled or resident',
         'A173': 'skilled employee or official',
         'A174': 'management or self-employed or highly qualified employee or officer'},
    18: {'A191': 'none',
         'A192': 'yes, registered under the customers name'},
    19: {'A201': 'yes',
         'A202': 'no'},
}

def map_values(row, col_index, mapping):
    """Map values of each row using the column index and dictionary mapping."""
    column_value = row.iloc[col_index]  # Use iloc to get the value at the column index
    if column_value in mapping:
        return mapping[column_value]
    return column_value

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Map CSV values to strings using predefined dictionaries.")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the input CSV file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output CSV file")

    # Parse arguments
    args = parser.parse_args()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(args.input_file)

    # Iterate through each column index and map values based on column index
    for col_index in range(len(df.columns)):  # Use range to iterate over column indices
        if col_index in value_dict:  # Check if the column index is in value_dict
            df.iloc[:, col_index] = df.apply(lambda row: map_values(row, col_index, value_dict[col_index]), axis=1)

    # Show the modified dataframe (optional)
    print("Transformed Data:")
    print(df.head())

    # Save the transformed DataFrame to the output file
    df.to_csv(args.output_file, index=False)
    print(f"Output saved to {args.output_file}")

if __name__ == "__main__":
    main()
