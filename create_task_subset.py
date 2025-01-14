import pandas as pd
import json
from copy import deepcopy
import os

csv_path = r"datasets\paraphrases-csv\par_gen_infer_all (1).csv"
destination_pth = r"datasets\paraphrased-sets"

par_sets = ['Male', 'Female', 'Ambiguous']
paraphrases_df = pd.read_csv(csv_path)
task_json = {}

print(paraphrases_df.head())

with open("siqa_task.json", mode='r') as f:
    task_json = json.load(f)
    
for set in par_sets:
    set_task_json = deepcopy(task_json)
    paraphrases_set = paraphrases_df[paraphrases_df['Category']==set].sort_values(by=['Row ID'])
    paraphrases_set = paraphrases_set[['Paraphrases']].reset_index(drop=True)
    # print(paraphrases_set)
    for i in range(len(set_task_json['examples'])):
        quest = set_task_json['examples'][i]['input'].split(".")[-1]
        set_task_json['examples'][i]['input'] = paraphrases_set.loc[i, 'Paraphrases'] + f" {quest}"
    print(set_task_json)
    with open(os.path.join(destination_pth, f"task_{set.split()[0]}.json"), mode='w') as f:
        json.dump(set_task_json, f)

