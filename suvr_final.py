import os
import nibabel as nib
import numpy as np

def MCSUVR(petfile, wmparcfile):
	### petfile is the path of pet image need to calculate MCSUVR:str
	### wmparcfile is the path of standard pet inage: str
	V_pet=nib.load(petfile).get_fdata() ##(w,h,c)
	np.nan_to_num(V_pet, copy=False, nan=0.0)  ###numpy
	##
	V_wmparc = nib.load(wmparcfile).get_fdata()  ###numpy
	np.nan_to_num(V_wmparc, copy=False, nan=0.0)
	###
	CBmask = np.zeros(V_pet.shape)
	CBWmask = np.array(CBmask, copy=False)
	MCmask = np.array(CBmask, copy=True)
	##

	##Reference Region ---Cerebellum cortex
	CBmask[V_wmparc == 8] = 1
	CBmask[V_wmparc == 47] = 1

	##Target Region -- Mean cortical Regions
	MCmask[V_wmparc == 1012] = 1 #lh-LOF (lateral-orbital-frontal)
	MCmask[V_wmparc == 2012] = 1 #rh-LOF
	MCmask[V_wmparc == 1014] = 1 #lh-MOF (medial-orbital-frontal)
	MCmask[V_wmparc == 2014] = 1 #rh-MOF
	MCmask[V_wmparc == 1015] = 1 #lh-MT  (middle-temporal)
	MCmask[V_wmparc == 2015] = 1 #rh-MT
	MCmask[V_wmparc == 1030] = 1 #lh-ST  (superior-temporal)
	MCmask[V_wmparc == 2030] = 1 #rh-ST
	MCmask[V_wmparc == 1025] = 1 #lh-PREC (precuneus)
	MCmask[V_wmparc == 2025] = 1 #rh-PREC
	MCmask[V_wmparc == 1027] = 1 #lh-RMF (rostral-middle-frontal)
	MCmask[V_wmparc == 2027] = 1 #rh-RMF
	MCmask[V_wmparc == 1028] = 1 #lh-SF  (superior-frontal)
	MCmask[V_wmparc == 2028] = 1 #rh-SF

	MC_suvr=np.mean(V_pet[MCmask==1])/np.mean(V_pet[CBmask==1])

	return MC_suvr

if __name__ == '__main__':
	data_split = '/data/hohokam/Yanxi/Data/mri2pet/adni/data_split.csv'
	pet_folder = '/data/hohokam/Yanxi/Data/mri2pet/adni/labels_256'
	wmparc_folder = '/data/hohokam/Yanxi/Data/mri2pet/adni/wmparc'
	new_data_split = '/data/hohokam/Yanxi/Data/mri2pet/adni/data_split_256.txt'

	with open(data_split) as f:
		f.readline()
		with open(new_data_split, 'w') as fout:
			fout.write(','.join(['mri_id', 'pet_id', 'age', 'sex', 'apoe', 'abeta', 'suvr', 'suvr_positivity', 'group']) + '\n')
			for line in f:
				linelist = line.split('\n')[0].split(',')
				mri_id = linelist[0]
				pet_id = linelist[1]
				age = linelist[2]
				sex = linelist[3]
				apoe = linelist[4]
				abeta = linelist[5]
				group = linelist[6]
				pet_file = os.path.join(pet_folder, pet_id)
				wmparc_file = os.path.join(wmparc_folder, mri_id.split('.')[0] + '.mgz')
				suvr = MCSUVR(pet_file, wmparc_file)
				suvr_positivity = '1' if suvr > 1.19 else '0'
				fout.write(','.join([mri_id, pet_id, age, sex, apoe, abeta, str(suvr), suvr_positivity, group]) + '\n')

	

	
	