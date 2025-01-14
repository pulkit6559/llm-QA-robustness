import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import gc

import torch
from xsbert import utils
from xsbert.models import XSMPNet, load_model

model_name = 'xs_distilroberta'
model = load_model(model_name, model_dir='../xs_models/')

# m_f_scores = pd.read_csv("../male_female_par_scores.csv")

m_labels = pd.read_csv("../baselines/base-male-all.csv")
f_labels = pd.read_csv("../baselines/base-female-all.csv")

m_f_df = pd.DataFrame({
    'idx': m_labels['idx'],
    'Original': m_labels['sentences_df1'],
    'Paraphrases_m': m_labels['sentences_df2'],
    'Paraphrases_f': f_labels['sentences_df2'],
    'choice_m': m_labels['scores_df2'],
    'choice_f': f_labels['scores_df2'],
    'choice_b': m_labels['scores_df1'],
    'target': m_labels['targets']
    
    # Add any other relevant columns
})

df_base_true = m_f_df

df_base_true['Original'] = df_base_true['Original'].apply(lambda x: " ".join(x.split(".")[:-1]))
df_base_true['Paraphrases_m'] = df_base_true['Paraphrases_m'].apply(lambda x: " ".join(x.split(".")[:-1]))
df_base_true['Paraphrases_f'] = df_base_true['Paraphrases_f'].apply(lambda x: " ".join(x.split(".")[:-1]))

xbert_attr_values_m_f = []
ra = []
rb = []
rr = []
attr_err = []

start_idx = int(sys.argv[1])

df_base_true_1000 = df_base_true.iloc[start_idx:start_idx+500]

for i, row in df_base_true_1000.iterrows():
    
    print(f"idx, Original: {i, row['Original']}")
    
    if i%20==0:
        del model
        gc.collect()
        model = load_model(model_name, model_dir='../xs_models/')
        model.to(torch.device('cuda:0'))
    model.reset_attribution()
    model.init_attribution_to_layer(idx=4, N_steps=15)
    # print(row)
    try:
        A_m, tokens_a_m, tokens_b_m, score_m, ra_m, rb_m, rr_m = model.attribute_prediction(
            row['Original'][:140], 
            row['Paraphrases_m'][:140], 
            move_to_cpu=False,
            compute_lhs=True
        )
        attr_err_m = torch.abs(A_m.sum() - score_m).item()
    except:
        A_m, tokens_a_m, tokens_b_m, score_m, ra_m, rb_m, rr_m = torch.zeros(1), None, None, 1, None, None, None
        attr_err_m = -1
        # del model
        # gc.collect()
        # model = load_model(model_name, model_dir='../xs_models/')
        # model.to(torch.device('cuda:0'))
        # model.reset_attribution()
        # model.init_attribution_to_layer(idx=4, N_steps=15)
        # torch.cuda.empty_cache()
        # a, b = row['Original'][:100], row['Paraphrases_m'][:100]
        # print(a,b)
        # A_m, tokens_a_m, tokens_b_m, score_m, ra_m, rb_m, rr_m = model.attribute_prediction(
        #     a, 
        #     b, 
        #     move_to_cpu=True,
        #     compute_lhs=True
        # )

    
    
    del A_m
    torch.cuda.empty_cache()
    
    try:
        A_f, tokens_a_f, tokens_b_f, score_f, ra_f, rb_f, rr_f = model.attribute_prediction(
        row['Original'][:140], 
        row['Paraphrases_f'][:140], 
        move_to_cpu=False,
        compute_lhs=True
        )
        attr_err_f = torch.abs(A_f.sum() - score_f).item()
    except:
        A_f, tokens_a_f, tokens_b_f, score_f, ra_f, rb_f, rr_f = torch.zeros(1), None, None, 1, None, None, None
        attr_err_f = -1
        # del model
        # gc.collect()
        # model = load_model(model_name, model_dir='../xs_models/')
        # model.to(torch.device('cuda:0'))
        # model.reset_attribution()
        # model.init_attribution_to_layer(idx=4, N_steps=15)
        # torch.cuda.empty_cache()
        # a, b = row['Original'][:100], row['Paraphrases_f'][:100]
        # print(a,b)
        # A_f, tokens_a_f, tokens_b_f, score_f, ra_f, rb_f, rr_f = model.attribute_prediction(
        #     a, 
        #     b, 
        # move_to_cpu=False,
        # compute_lhs=True
        # )
        

    
    
    del A_f
    torch.cuda.empty_cache()
    
    print(f"Paraphrase_m: { row['Paraphrases_m']}")
    print(f"Attribution score O-M: {score_m}")
    print(f"Paraphrase_f: { row['Paraphrases_f']}")
    print(f"Attribution score O-F: {score_f}")
    xbert_attr_values_m_f.append([score_m, score_f])
    
    print(f"ra, rb, rr: {[ra_m, ra_f]}, {[rb_m, rb_f]}, {[rr_m, rr_f]}")
    print(f"attr_error: {[attr_err_m, attr_err_f]}")
    ra.append([ra_m, ra_f])   
    rb.append([rb_m, rb_f])   
    rr.append([rr_m, rr_f])   
    attr_err.append([attr_err_m, attr_err_f]) 

attr_err = np.array(attr_err)
xbert_attr_values_m_f = np.array(xbert_attr_values_m_f)

df_base_true_1000['attr_error_m'] = attr_err[:,0]
df_base_true_1000['attr_error_f'] = attr_err[:,1]
df_base_true_1000['model_score_m'] = xbert_attr_values_m_f[:,0]
df_base_true_1000['model_score_f'] = xbert_attr_values_m_f[:,1]

df_base_true_1000.to_csv(f"m_f_par_scores_xsbert_{start_idx}_{start_idx+500}.csv")


