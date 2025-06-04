import os
import shutil
import numpy as np
import pandas as pd

def suvr_stat():
    with open('suvr_stat.csv', 'w') as f:
        f.write('exp,mean_gen,std_gen,mean_true,std_true\n')
        for fname in os.listdir('../results/suvr_ptau_lp_mid'):
            df = pd.read_csv(os.path.join('../results/suvr_ptau_lp_mid', fname))
            name = fname.split('.')[0]
            gen_suvr = np.array(df['gen_suvr'])
            true_suvr = np.array(df['true_suvr'])
            m_gen = np.mean(gen_suvr)
            std_gen = np.std(gen_suvr)
            m_true = np.mean(true_suvr)
            std_true = np.std(true_suvr)
            f.write(','.join([name, str(m_gen), str(std_gen), str(m_true), str(std_true)]) + '\n')


if __name__ == '__main__':
    suvr_stat()