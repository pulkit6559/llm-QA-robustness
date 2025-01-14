import pandas as pd
import os

def split_csv(input_file, output_folder, num_splits=5):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Calculate the number of rows in each split
    rows_per_split = len(df) // num_splits

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    curr_idx = 0
    # Split the dataframe into 5 parts and save each part as a separate CSV file
    for i in range(num_splits):
        start_index = curr_idx
        end_index = curr_idx + rows_per_split
        if end_index>len(df):
            end_index = len(df)-1
        while "---------" not in df.iloc[end_index]['Orginal']:
            end_index+=1
        split_df = df.iloc[start_index:end_index+1]
        
        # Save the split dataframe to a CSV file
        output_file = os.path.join(output_folder, f'split_{i + 1}.csv')
        split_df.to_csv(output_file, index=False)
        print(f'Split {i + 1} saved to {output_file}')
        curr_idx = end_index+1


if __name__ == "__main__":

    paraphrase_csv_dir = "datasets/paraphrases-csv"
    output_folder = os.path.join(paraphrase_csv_dir, "split")
    in_file = os.path.join(paraphrase_csv_dir, 'sentences.csv')

    # Number of splits (in this case, 15)
    num_splits = 15

    # Call the function to split the CSV
    split_csv(in_file, output_folder, num_splits)
