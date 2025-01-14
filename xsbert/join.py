import pandas as pd
import os

joined_df = pd.DataFrame()

for f in os.listdir("."):
    if f.endswith(".csv"):
        joined_df = pd.concat([joined_df, pd.read_csv(f)], axis=0, ignore_index=True)
        print(joined_df.head())

joined_df = joined_df.iloc[:, 1:]
print(joined_df)
joined_df.to_csv("m_f_scores_1954_xsbert.csv", index=False)