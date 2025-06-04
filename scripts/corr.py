import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def corr(logname):
	gen_suvr = []
	true_suvr = []
	with open(logname) as f:
		f.readline()
		for line in f:
			linelist = line.split('\n')[0].split(',')
			gen_suvr.append(float(linelist[2]))
			true_suvr.append(float(linelist[3]))

	gen_suvr = np.array(gen_suvr)
	true_suvr = np.array(true_suvr)

	return pearsonr(gen_suvr, true_suvr)

def pix_corr(gen_folder, true_folder, log_name):
	gen_suvr = []
	true_suvr = []
	with open(os.path.join(root, log_name + '.txt')) as f:
		f.readline()
		for line in f:
			linelist = line.split('\n')[0].split(',')
			gen_suvr.append(float(linelist[0]))
			true_suvr.append(float(linelist[1]))

	gen_suvr = np.array(gen_suvr)
	true_suvr = np.array(true_suvr)

	print(log_name + ': ', pearsonr(gen_suvr, true_suvr))

def regional_suvr_corr(log_file):
	region_list = ['lh-LOF', 'rh-LOF', 'lh-MOF', 'rh-MOF', 'lh-MT', 'rh-MT', 'lh-ST', 'rh-ST', 'lh-PREC', 'rh-PREC', 'lh-RMF', 'rh-RMF', 'lh-SF', 'rh-SF']
	df = pd.read_csv(log_file)
	# print(df.columns)
	# exit()

	r_list = []
	p_list = []

	for region in region_list:
		gen_arry = df[region + '_gen'].to_numpy()
		true_arry = df[region + '_true'].to_numpy()
		r, p = pearsonr(gen_arry, true_arry)
		r_list.append(r)
		p_list.append(p)

	assert len(r_list) == len(p_list) == len(region_list)

	return r_list, p_list

def suvr_mse(log_file):
	gen_suvr = []
	true_suvr = []
	with open(log_file) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')
			gen_suvr.append(float(linelist[2]))
			true_suvr.append(float(linelist[3]))

	gen_suvr = np.array(gen_suvr)
	true_suvr = np.array(true_suvr)

	suvr_diff = np.dot(np.subtract(gen_suvr, true_suvr), np.subtract(gen_suvr, true_suvr))
	mse = suvr_diff / len(gen_suvr)

	return mse

def suvr_cls(logfile, data_split):
	pos_map = {}
	df = pd.read_csv(data_split)
	for i in range(len(df)):
		mri = df.loc[i, 'mri_id']
		positivity = df.loc[i, 'positivity']
		pos_map[mri] = positivity

	suvrs = []
	labels = []

	with open(logfile) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')
			mri = linelist[0].split('.')[0]
			suvrs.append(float(linelist[2]))
			labels.append(pos_map[mri])

	X = np.array(suvrs).reshape(-1, 1)
	y = np.array(labels)

	# X_r, y_r = SMOTE().fit_resample(X, y)

	# X, y = X_r, y_r

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
		# y_pred = model.predict(X_test)
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

	return accuracy, precision, recall, f1, auc

def suvr_log_cls(logfile):
	suvrs = []
	labels = []

	with open(logfile) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')

			suvrs.append(float(linelist[2]))
			labels.append(1 if float(linelist[3]) > 1.19 else 0)

	X = np.array(suvrs).reshape(-1, 1)
	y = np.array(labels)

	# X_r, y_r = SMOTE().fit_resample(X, y)

	# X, y = X_r, y_r

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
		# y_pred = model.predict(X_test)
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

	return accuracy, precision, recall, f1, auc

def suvr_log_cls_good(fname):
	suvrs = []
	labels = []
	with open(fname) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')

			suvrs.append(float(linelist[2]))
			labels.append(1 if float(linelist[3]) > 1.1 else 0)
	
	X = np.array(suvrs).reshape(-1, 1)
	y = np.array(labels)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

	fit = LogisticRegression().fit(X_train, y_train)
	y_pred = fit.predict(X_test)
	# cm = metrics.confusion_matrix(y_test, y_pred)
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	# auc = roc_auc_score(y_test, fit.predict_proba(X_test)[:, 1])

	return accuracy, precision, recall, f1

def suvr_corr_plot(log_file, exp_name, plot_name, out_folder):
	gen_suvr = []
	true_suvr = []
	with open(log_file) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')
			gen_suvr.append(float(linelist[2]))
			true_suvr.append(float(linelist[3]))

	x = np.array(gen_suvr)
	y = np.array(true_suvr)

	slope, intercept = np.polyfit(x, y, 1)
	regression_line = slope * x + intercept

	plt.scatter(x, y, color='blue', label='Data points')
	plt.plot(x, regression_line, color='black', label='Regression line')

	plt.xlabel('gen_SUVR')
	plt.ylabel('true_SUVR')
	plt.title(plot_name)

	plt.savefig(os.path.join(out_folder, exp_name + '.png'))

def auc_plot(name, fname, output):
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve, roc_auc_score
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import make_classification
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier

	suvrs = []
	labels = []
	with open(fname) as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')
			suvrs.append(float(linelist[2]))
			labels.append(1 if float(linelist[3]) > 1.19 else 0)
	
	X = np.array(suvrs).reshape(-1, 1)
	y = np.array(labels)

	group_len = len(y) // 3 + 1

	y_prob = []

	for i in range(3):
		X_test = X[group_len * i:min(group_len * (i + 1), len(y))]
		X_train = np.concat((X[:group_len * i], X[min(group_len * (i + 1), len(y)-1):]))
		y_test = y[group_len * i:min(group_len * (i + 1), len(y))]
		y_train = np.concat((y[:group_len * i], y[min(group_len * (i + 1), len(y)-1):]))

		model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
		y_pred_proba = model.predict_proba(X_test)[:, 1]
		y_prob += y_pred_proba.tolist()

	y_prob = np.array(y_prob)
	# Plot ROC curves
	plt.figure(figsize=(8, 6))

	# y_pred_proba = model.predict_proba(X_test)[:, 1]
	fpr, tpr, _ = roc_curve(y, y_prob)
	auc = roc_auc_score(y, y_prob)
	plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC pTau-217')
	plt.legend()
	plt.savefig(output)

def auc_plot_folder(folder, output):
	import matplotlib.pyplot as plt
	from sklearn.metrics import roc_curve, roc_auc_score
	from sklearn.model_selection import train_test_split
	from sklearn.datasets import make_classification
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier

	plt.figure(figsize=(8, 6))
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC Abeta')

	# names = ['pix2pix', 'pix2pix_ptau_c2n', 'pix2pix_ptau_c2n', 'cyclegan', 'cyclegan_ptau_c2n', 'cyclegan_ptau_c2n', 'sharegan', 'sharegan_ptau_c2n', 'sharegan_ptau_c2n']
	names = ['pix2pix', 'pix2pix_abeta', 'cyclegan', 'cyclegan_abeta', 'sharegan', 'sharegan_abeta']

	for name in names:
		suvrs = []
		labels = []
		with open(os.path.join(folder, name + '.txt')) as f:
			f.readline()
			for line in f:
				linelist = line.strip().split(',')
				suvrs.append(float(linelist[2]))
				labels.append(1 if float(linelist[3]) > 1.19 else 0)
		
		X = np.array(suvrs).reshape(-1, 1)
		y = np.array(labels)

		X_r, y_r = SMOTE().fit_resample(X, y)

		X, y = X_r, y_r

		group_len = len(y) // 3 + 1

		y_prob = []

		for i in range(3):
			X_test = X[group_len * i:min(group_len * (i + 1), len(y))]
			X_train = np.concat((X[:group_len * i], X[min(group_len * (i + 1), len(y)-1):]))
			y_test = y[group_len * i:min(group_len * (i + 1), len(y))]
			y_train = np.concat((y[:group_len * i], y[min(group_len * (i + 1), len(y)-1):]))

			model = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
			y_pred_proba = model.predict_proba(X_test)[:, 1]
			y_prob += y_pred_proba.tolist()

		y_prob = np.array(y_prob)
		# Plot ROC curves

		# y_pred_proba = model.predict_proba(X_test)[:, 1]
		fpr, tpr, _ = roc_curve(y, y_prob)
		auc = roc_auc_score(y, y_prob)
		plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

		plt.plot([0, 1], [0, 1], 'k--')
		plt.legend()
		
	plt.savefig(output)

if __name__ == '__main__':
	bbbm = 'ptau'
	data_split = '/data/hohokam/Yanxi/Data/mri2pet/adni/data_split_ptau.csv'
	suvr_folder = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_ptau'
	plot_folder = '/scratch/ychen855/mri2pet/CycleGAN_3D/cuvr_cor_plots'
	pixcorr_folder = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/pixcorr'
	regional_suvr_folder = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/regional_suvr'
	out_regional_suvr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/regional_suvr_corr.csv'
	out_suvr_mse = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_mse.csv'
	out_suvr_corr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_corr_ptau.csv'
	out_suvr_acc = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_corr_ptau.csv'
	out_plot = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_corr_plot'
	which_epochs = [60, 80, 100, 120]
	n_epochs = [120]

	# names = ['cyclegan', 'cyclegan_image', 'cyclegan_add' 'cyclegan_concat',
	# 		'pix2pix', 'pix2pix_image', 'pix2pix_add', 'pix2pix_concat',
	# 		'sharegan', 'sharegan_image', 'sharegan_add', 'sharegan_concat']
	# names = ['cyclegan', 'cyclegan_ptau_c2n_concat', 'cyclegan_ptau_c2n_concat',
    #         'pix2pix', 'pix2pix_ptau_c2n_concat', 'pix2pix_ptau_c2n_concat',
    #         'sharegan', 'sharegan_ptau_c2n_concat', 'sharegan_ptau_c2n_concat']

	exp_names = ['cyclegan_ptau_c2n_mid', 'cyclegan_c_ptau_c2n_mid',
				'pix2pix_ptau_c2n_mid', 'pix2pix_c_ptau_c2n_mid'
				'sharegan_ptau_c2n_mid', 'sharegan_c_ptau_c2n_mid']

	# exp_names = ['cyclegan_80_100', 'abeta_concat_60_80', 'pix2pix_20_20', 'pix2pix_abeta_concat_20_20', 'sharegan_100_120', 'sharegan_concat_120_120']

	# SUVR corr
	with open('../results/suvr_corr_ptau.csv', 'w') as f:
		f.write('exp,r,pval\n')
		for fname in sorted(os.listdir('../results/suvr_ptau')):
		# for exp_name in exp_names:
			name = fname.split('.')[0]
			logname = os.path.join(suvr_folder, name + '.txt')
			pearson = corr(logname)
			f.write(','.join([name, str(pearson.statistic), str(pearson.pvalue)]) + '\n')
			

	# # regional SUVR corr
	# region_list = ['lh-LOF', 'rh-LOF', 'lh-MOF', 'rh-MOF', 'lh-MT', 'rh-MT', 'lh-ST', 'rh-ST', 'lh-PREC', 'rh-PREC', 'lh-RMF', 'rh-RMF', 'lh-SF', 'rh-SF']
	# with open(out_regional_suvr, 'w') as f:
	# 	f.write(','.join(['model'] + [ele + '_r,' + ele + '_p' for ele in region_list]) + '\n')

	# 	for exp_name in exp_names:
	# 		for which_epoch in which_epochs:
	# 			for n_epoch in n_epochs:
	# 				logname = os.path.join(regional_suvr_folder, exp_name + '_' + str(which_epoch) + '_' + str(n_epoch) + '.txt')
	# 				r_list, p_list = regional_suvr_corr(logname)
	# 				f.write(','.join([exp_name] + [str(r_list[i]) + ',' + str(p_list[i]) for i in range(len(r_list))]) + '\n')

	# SUVR mse
	# with open(out_suvr_mse, 'w') as f:
	# 	f.write('model,mse_suvr\n')
	# 	for exp_name in exp_names:
	# 		for which_epoch in which_epochs:
	# 			for n_epoch in n_epochs:
	# 				logname = os.path.join(suvr_folder, exp_name + '_' + str(which_epoch) + '_' + str(n_epoch) + '.txt')
	# 				mse = suvr_mse(logname)
	# 				f.write(exp_name + ',' + str(mse) + '\n')


	# SUVR cls
	# with open('../results/suvr_acc_ptau_c2n_mid.csv', 'w') as f:
	# 	f.write('model,accuracy,precision,recall,f1,auc\n')
	# 	for fname in sorted(os.listdir('../results/suvr_ptau_c2n_mid')):
	# 		name = fname.split('.')[0]
	# 		logname = os.path.join('../results/suvr_ptau_c2n_mid', fname)

	# 		accuracy, precision, recall, f1, auc = suvr_log_cls(logname)
	# 		f.write(name + ',' + str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(auc) +  '\n')


	# SUVR cls true
	data_split_true = '/scratch/ychen855/Data/mri2pet/adni_ptau/data_split_ptau.csv'
	with open('../results/suvr_acc_ptau.csv', 'w') as f:
		f.write('model,accuracy,precision,recall,f1,auc\n')
		for fname in sorted(os.listdir('../results/suvr_ptau')):
			name = fname.split('.')[0]
			logname = os.path.join('../results/suvr_ptau', fname)

			accuracy, precision, recall, f1, auc = suvr_cls(logname, data_split_true)
			f.write(name + ',' + str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + ',' + str(auc) +  '\n')


	# exp_names = ['1338', 'abeta_1338', 'pix2pix', 'sharegan']

	# # SUVR corr plot
	# exp_name = 'sharegan'
	# plot_name = 'ShareGAN'
	# for which_epoch in which_epochs:
	# 	for n_epoch in n_epochs:
	# 		logname = os.path.join(suvr_folder, exp_name + '_' + str(which_epoch) + '_' + str(n_epoch) + '.txt')
	# 		suvr_corr_plot(logname, exp_name, plot_name, out_plot)
	
	# SUVR cls good
	# with open('../results/suvr_final.csv', 'w') as f:
	# 	f.write('model,accuracy,precision,recall,f1\n')
	# 	for name in names:
	# 		accuracy, precision, recall, f1, _ = suvr_log_cls(os.path.join('../results/suvr_ptau', name + '.txt'))
	# 		f.write(name + ',' + str(accuracy) + ',' + str(precision) + ',' + str(recall) + ',' + str(f1) + '\n')

	# auc plot
	# for fname in os.listdir('../results/results_plot'):
	# 	exp_name = fname.split('.')[0]
	# 	auc_plot(exp_name, os.path.join('../results/suvr_ptau', fname), os.path.join('../results/auc_plot', exp_name))
	# 	exit()

	# auc_plot_folder('../results/results_plot', '../results/plot_abeta.pdf')