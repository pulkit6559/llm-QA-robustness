import csv

# Open the CSV file for reading
with open('m_f_par_scores_xsbert.csv', 'r', newline='') as file:
    reader = csv.DictReader(file)

    # Define new headers for the split columns
    new_headers = ['attr_error_1', 'attr_error_2', 'model_score_1', 'model_score_2']

    # Create a new CSV file for writing the modified data
    with open('modified_file.csv', 'w', newline='') as modified_file:
        writer = csv.DictWriter(modified_file, fieldnames=[*reader.fieldnames, *new_headers])
        writer.writeheader()

        # Iterate over each row in the original CSV file
        for row in reader:
            # Split the values in "attr_error" column
            attr_error_values = eval(row['attr_error'])
            row['attr_error_1'] = attr_error_values[0]
            row['attr_error_2'] = attr_error_values[1]

            # Split the values in "model_score" column
            model_score_values = eval(row['model_score'])
            row['model_score_1'] = model_score_values[0]
            row['model_score_2'] = model_score_values[1]

            # Write the modified row to the new CSV file
            del row['attr_error']
            del row['model_score']
            writer.writerow(row)
