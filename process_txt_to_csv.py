import re
import pandas as pd
import os
from cleantext import clean

paraphrase_dir = "datasets/paraphrases-llm"
paraphrase_csv_dir = "datasets/paraphrases-csv"
in_file = 'par_gender_temp06_rp11.txt'

# Read the text file
with open(os.path.join(paraphrase_dir, in_file), 'r', encoding="utf-8", errors="replace") as file:
    text = file.read()

# Split the text into sections using '---------------'
sections = re.split('-{15,}', text)[1:]

def clean_sentence(sentence):
    # remove any kind of quotes from the sentence
    sentence = sentence.strip('"').strip("'")
    # remove any number followed by a dot and a space from the sentence
    sentence = re.sub(r"\d+\. ", "", sentence)

    return sentence


# Define a function to extract sentences and paraphrases
def extract_paraphrase_MF(section):
    sentences = []
    paraphrases = []

    # Use regex to find Sentence: and Paraphrase: lines
    sentence_pattern = r"Sentence:\s+(.+)"
    paraphrase_pattern = r"(\d+\.\s+(Male:|Female:|Ambiguous:)\s+(.+))"

    for line in section.split('\n'):
        sentence_match = re.match(sentence_pattern, line)
        paraphrase_match = re.match(paraphrase_pattern, line)

        if sentence_match:
            sentences.append(sentence_match.group(1).strip())
        elif paraphrase_match:
            print(paraphrase_match.group(1))
            paraphrases.append(paraphrase_match.group(1).strip().removeprefix("<p>").removesuffix("</p>"))

    return sentences, paraphrases

def extract_paraphrase_AB(section):
    sentences = []
    paraphrases = []
    print('-'*15)
    # Use regex to find Sentence: and Paraphrase: lines
    sentence_pattern = r"Sentence\s*:\s*\n?(.*?)\n"


    # for line in section.split('\n'):
    sentence_match = re.findall(sentence_pattern, section)
    
    try:
        paraphrase_pattern_A = r'Male\s*\(.+\)?:\s*\n?(.*?)\n'
        # r"A \((.+?)\):\s*(.+?)(?=\n)"
        paraphrase_pattern_B = r'Female\s*\(.+\)?:\s*\n?(.*?)\n'
        # r"B \((.+?)\):\s*(.+?)(?=\n)"
        paraphrase_pattern_AMB = r'Ambiguous\s*\(.+\)?:\s*\n?(.*?)\n'
        # r"Ambiguous \((.+?)\):\s*(.+?)(?=\n)"
        
        paraphrase_match_A = re.findall(paraphrase_pattern_A, section)[0]
        paraphrase_match_B = re.findall(paraphrase_pattern_B, section)[0]
        paraphrase_match_AMB = re.findall(paraphrase_pattern_AMB, section)[0]
        
        print("A: ", paraphrase_match_A)
        print("B: ", paraphrase_match_B)
        print("AMB: ", paraphrase_match_AMB)
        
    except:
        print("HERE")
        paraphrase_pattern_A = r"Male\s*(\(.*\))*:\s*\n?(.*?)\n"
        paraphrase_pattern_B = r"Female\s*(\(.*\))*:\s*\n?(.*?)\n"
        paraphrase_pattern_AMB = r"Ambiguous\s*(\(.*\))*:\s*\n?(.*?)\n"
        paraphrase_match_A = re.findall(paraphrase_pattern_A, section)[0][1]
        paraphrase_match_B = re.findall(paraphrase_pattern_B, section)[0][1]
        paraphrase_match_AMB = re.findall(paraphrase_pattern_AMB, section)[0][1]
        
        print("A: ", paraphrase_match_A)
        print("B: ", paraphrase_match_B)
        print("AMB: ", paraphrase_match_AMB)
          
    paraphrase_match = [paraphrase_match_A, paraphrase_match_B, paraphrase_match_AMB]

    for p in paraphrase_match:
        print(sentence_match)
        sentences.append(sentence_match)
    # elif paraphrase_match:
    #     print(paraphrase_match)
        # paraphrases.append(paraphrase_match.group(1).strip().removeprefix("<p>").removesuffix("</p>"))

    return sentences, paraphrase_match

# Process each section
data = {'Row ID': [], 'Paraphrases': [], 'Category': [], 'Orginal': []}

idx = 0
# Process each section
for section_index, section in enumerate(sections):
    # sentences, paraphrases = extract_paraphrase_MF(section)
    try:
        sentences, paraphrases = extract_paraphrase_AB(section)
    except:
        sentence_pattern = r"Sentence\s*:\s*\n?(.*?)\n"
        # for line in section.split('\n'):
        sentence_match = re.findall(sentence_pattern, section)
        sentences = [sentence_match]
        paraphrases = [sentence_match[0] for i in range(3)]

    if paraphrases:
        for paraphrase_index, paraphrase in enumerate(paraphrases, start=1):
            if paraphrase_index==1:
                data['Orginal'].append(clean_sentence(sentences[0][0]))
            else:
                data['Orginal'].append(clean_sentence(sentences[0][0]))
            # data['Orginal'].append(sentences[0])
            data['Row ID'].append(f"{idx}")
            
            if paraphrase_index==1:
                category = 'Male'
            elif paraphrase_index==2:
                category = 'Female'
            elif paraphrase_index==3:
                category = 'Ambiguous'
            # print(paraphrase)
            # paraphrase = paraphrase.split(":")[1].strip().strip("\"").strip("</p>")
            # print(paraphrase)
            if paraphrase=="":
                paraphrase = '<nostr>'
            data['Paraphrases'].append(clean_sentence(paraphrase))
            data['Category'].append(category)
            idx+=1

        # Add an empty row after each group
        data['Row ID'].append(f"{idx}")
        data['Paraphrases'].append('-'*15)
        data['Category'].append('-'*15)
        data['Orginal'].append('-'*15)
        idx+=1

# Create a DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
save_file = os.path.join(paraphrase_csv_dir, in_file.split(".")[0]+".csv")
df.to_csv(save_file, index=False)
# This script will create a CSV file named "paraphrases.csv" containing row IDs and sentences, with an empty row after each group of paraphrases.


# sent_save_file = os.path.join(paraphrase_csv_dir, "sentences.csv")
# del data['Paraphrases']
# del data['Category']

# df = pd.DataFrame(data)
# df.to_csv(sent_save_file, index=False)




