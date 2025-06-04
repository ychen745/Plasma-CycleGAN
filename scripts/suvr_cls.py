import os
import shutil
import numpy as np
import pandas as pd
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

def suvr_cls():
	data_split = '/scratch/ychen855/Data/mri2pet/adni_ptau/data_split_ptau_lp_mid.csv'
	df = pd.read_csv(data_split)
	ptau_map = {}
	pos_map = {}
	for i in range(len(df)):
		ptau_map[df.loc[i, 'mri_id'] + '.nii'] = df.loc[i, 'ptau_lp']
		pos_map[df.loc[i, 'mri_id'] + '.nii'] = df.loc[i, 'positivity']

	result = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_ptau_lp_mid/cyclegan_c_ptau_lp_mid_120_120.txt'
	df_res = pd.read_csv(result)

	ptau = []
	pos = []
	suvr = []

	for i in range(len(df_res)):
		suvr.append(df_res.loc[i, 'gen_suvr'])
		mri_id = df_res.loc[i, 'mri']
		pos.append(pos_map[mri_id])
		ptau.append(ptau_map[mri_id])
	
	# ptau = np.array(ptau)
	# suvr = np.array(suvr)
	# pos = np.array(pos)

	# X = np.array(ptau).reshape(-1, 1)
	X = np.concat((np.array(suvr).reshape(-1, 1), np.array(ptau).reshape(-1, 1)), axis=1)
	# print(X.shape)
	# exit()
	y = np.array(pos)

	# print(len(X))
	# print(len(y))

	# exit()

	idx_list = [i for i in range(len(y))]
	random.seed(42)
	random.shuffle(idx_list)

	X = X[idx_list]
	y = y[idx_list]

	group_len = len(y) // 5 + 1
	y_pred_list = []
	y_prob_list = []

	for i in range(5):
		X_test = X[group_len * i:min(group_len * (i + 1), len(y))]
		X_train = np.concat((X[:group_len * i], X[min(group_len * (i + 1), len(y)-1):]))
		y_test = y[group_len * i:min(group_len * (i + 1), len(y))]
		y_train = np.concat((y[:group_len * i], y[min(group_len * (i + 1), len(y)-1):]))

		model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)

		y_pred_proba = model.predict_proba(X_test)[:, 1]

		threshold = 0.5
		y_pred = np.where(y_pred_proba >= threshold, 1, 0)

		y_prob_list += y_pred_proba.tolist()
		y_pred_list += y_pred.tolist()

	y_prob_list = np.array(y_prob_list)
	y_pred_list = np.array(y_pred_list)

	accuracy = accuracy_score(y, y_pred_list)
	precision = precision_score(y, y_pred_list)
	recall = recall_score(y, y_pred_list)
	f1 = f1_score(y, y_pred_list)
	auc = roc_auc_score(y, y_prob_list)

	print(auc)

if __name__ == '__main__':
	suvr_cls()
