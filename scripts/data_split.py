import os
import shutil
import pandas as pd
import random

df = pd.read_csv('/scratch/ychen855/Data/mri2pet/adni_ptau/data_split_ptau_c2n_mid_split.csv')

df = df[df['ptau_c2n_group'] == 1]
for i in range(len(df)):
    df.loc[i, 'group'] = random.randint(0, 9)

df.to_csv('/scratch/ychen855/Data/mri2pet/adni_ptau/data_split_ptau_c2n_mid.csv', index=False)