import re
import pandas as pd

# Read the text file
with open('paraphrase_out.txt', 'r') as file:
    text = file.read()

# Split the text into sections using '---------------'
sections = re.split('-{15,}', text)

# Define a function to extract sentences and paraphrases
def extract_sentences_and_paraphrases(section):
    sentences = []
    paraphrases = []

    # Use regex to find Sentence: and Paraphrase: lines
    sentence_pattern = r"Sentence:\s+(.+)"
    paraphrase_pattern = r"\d+\.\s+(.+)"

    for line in section.split('\n'):
        sentence_match = re.match(sentence_pattern, line)
        paraphrase_match = re.match(paraphrase_pattern, line)

        if sentence_match:
            sentences.append(sentence_match.group(1).strip())
        elif paraphrase_match:
            paraphrases.append(paraphrase_match.group(1).strip().removeprefix("<p>").removesuffix("</p>"))

    return sentences, paraphrases

# Process each section
data = {'Row ID': [], 'Category': [], 'Paraphrases': [], 'Orginal': []}

idx = 0
# Process each section
for section_index, section in enumerate(sections):
    sentences, paraphrases = extract_sentences_and_paraphrases(section)

    if paraphrases:
        for paraphrase_index, paraphrase in enumerate(paraphrases, start=1):
            if paraphrase_index==1:
                data['Orginal'].append(sentences)
            else:
                data['Orginal'].append("")
            data['Row ID'].append(f"{idx}")
            category = paraphrase.split(":")[0].strip()
            paraphrase = paraphrase.split(":")[1].strip().strip("\"").strip("</p>")
            print(paraphrase)
            if paraphrase=="":
                paraphrase = '<nostr>'
            data['Paraphrases'].append(paraphrase)
            data['Category'].append(category)
            idx+=1

        # Add an empty row after each group
        data['Row ID'].append(f"{idx}")
        data['Paraphrases'].append('-'*15)
        data['Category'].append('-'*15)
        data['Orginal'].append("")
        idx+=1

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('paraphrases_cat.csv', index=False)
# This script will create a CSV file named "paraphrases.csv" containing row IDs and sentences, with an empty row after each group of paraphrases.



# !python train_tempobert.py  --model_name_or_path bert-base-uncased \
#     --train_path /datasets/nyt_with10k_every10  \
#     --do_train \
#     --output_dir output \
#     --time_embedding_type prepend_token \
#     --time_mlm_probability 0.9 \
#     --max_seq_length 128


