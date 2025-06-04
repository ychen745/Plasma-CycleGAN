import os
import shutil
import gzip
import nibabel as nib
from pathlib import Path

def move_fs():
	dest_root = '/scratch/ychen855/Data/mri2pet'
	src_root = '/data/amciilab/jay/datasets/mrpetsyn'
	dest_mri = os.path.join(dest_root, 'mri')
	dest_pet = os.path.join(dest_root, 'pet')

	f_id_map = open(os.path.join(dest_root, 'fid_map.csv'))
	id_map = dict()
	for line in f_id_map:
		linelist = line[:-1].split(',') if line[-1] == '\n' else line.split(',')
		id_map[linelist[0]] = linelist[1]

	for k, v in id_map.items():
		folder_mri = k.split('.')[0]
		folder_pet = v.split('_')[0]
		os.mkdir(os.path.join(dest_mri, 'fs', folder_mri))
		os.mkdir(os.path.join(dest_mri, 'fs', folder_mri, folder_mri))
		shutil.copy(os.path.join(src_root, 'FS7', folder_mri, 'surf', 'lh.pial'), os.path.join(dest_mri, 'fs', folder_mri, folder_mri))
		shutil.copy(os.path.join(src_root, 'FS7', folder_mri, 'surf', 'lh.white'), os.path.join(dest_mri, 'fs', folder_mri, folder_mri))
		shutil.copy(os.path.join(src_root, 'FS7', folder_mri, 'surf', 'rh.pial'), os.path.join(dest_mri, 'fs', folder_mri, folder_mri))
		shutil.copy(os.path.join(src_root, 'FS7', folder_mri, 'surf', 'rh.white'), os.path.join(dest_mri, 'fs', folder_mri, folder_mri))
		
		os.mkdir(os.path.join(dest_pet, 'nifti', folder_pet))
		shutil.copy(os.path.join(src_root, 'PUP_FBP', folder_pet, 'pet_proc', v), os.path.join(dest_pet, 'nifti', folder_pet))


def prepare_adni(pet_list, warped=False):
	dest_root = '/scratch/ychen855/Data/mri2pet/adni_ptau'
	src_root = '/data/amciilab/processedDataset/ADNI'

	mri_root = os.path.join(src_root, 'ADNI-FS')
	pet_root = os.path.join(src_root, 'ADNI-PUP', 'FBP')

	fdict = dict()
	# id_set = set()
	for fname in os.listdir(pet_root):
		if fname in pet_list:
			# pid = fname[fname.find('FBP') + 3:fname.find('L')]
			finfo = os.path.join(pet_root, fname, fname + '_pet.param')
			with open(finfo) as f:
				for line in f:
					if line.startswith('fsdir'):
						mri_id = line.split('/')[-2] if fname.startswith('FBP') else line.split('/')[-4]
						fdict[fname] = mri_id

	with open(os.path.join(dest_root, 'id_map.csv'), 'w') as f:
		for pet in pet_list:
			pet_gz = False
			if warped:
				pass
			else:
				exist = False
				for fmri in os.listdir(os.path.join(mri_root, fdict[pet], 'cat12')):
					if fmri.startswith(fdict[pet]):
						src_mri = os.path.join(mri_root, fdict[pet], 'cat12', fmri)
						src_wmparc = os.path.join(mri_root, fdict[pet], 'mri', 'wmparc.mgz')
						if fmri in os.listdir(os.path.join(dest_root, 'images_256')):
							exist = True
						break

				if pet + '_SUVR.nii' in os.listdir(os.path.join(pet_root, pet, 'pet_proc')):
					src_pet = os.path.join(pet_root, pet, 'pet_proc', pet + '_SUVR.nii')
				else:
					src_pet = os.path.join(pet_root, pet, 'pet_proc', pet + '_SUVR.nii.gz')
					pet_gz = True

				dest_mri = os.path.join(dest_root, 'images_256')
				dest_pet = os.path.join(dest_root, 'labels_256')
				dest_wmparc = os.path.join(dest_root, 'wmparc_256', fdict[pet] + '.mgz')

				f.write(fdict[pet] + 'N.nii,' + pet + '_SUVR.nii\n')

				if not exist:
					try:
						shutil.copy(src_mri, dest_mri)
					except:
						print('mri missing:', fdict[pet])

					try:
						shutil.copy(src_pet, dest_pet)
					except:
						print('pet missing:', fdict[pet])

					try:
						shutil.copy(src_wmparc, dest_wmparc)
					except:
						print('wmparc missing:', fdict[pet])

def check_dims(root, output):
	with open(output, 'w') as f:
		for fname in os.listdir(root):
			image = nib.load(os.path.join(root, fname)).get_fdata()
			f.write(fname + ',' + str(image.shape) + '\n')

def prepare_adni_split(pet_list, data_split):
	dest_root = '/scratch/ychen855/Data/mri2pet/adni/images_256'
	src_root = '/data/amciilab/processedDataset/ADNI'

	mri_root = os.path.join(src_root, 'ADNI-FS')
	pet_root = os.path.join(src_root, 'ADNI-PUP', 'FBP')

	fdict = dict()
	# id_set = set()
	for fname in os.listdir(pet_root):
		if fname in pet_list:
			# pid = fname[fname.find('FBP') + 3:fname.find('L')]
			finfo = os.path.join(pet_root, fname, fname + '_pet.param')
			with open(finfo) as f:
				for line in f:
					if line.startswith('fsdir'):
						mri_id = line.split('/')[-2]
						fdict[fname] = mri_id
				# id_set.add(pid)

	with open(os.path.join(dest_root, 'id_map.csv'), 'w') as f:
		for pet in pet_list:
			pet_gz = False
			
			if fdict[pet] + 'N.nii' in os.listdir(os.path.join(mri_root, fdict[pet], 'cat12')):
				src_mri = os.path.join(mri_root, fdict[pet], 'cat12', fdict[pet] + 'N.nii')
			else:
				src_mri = os.path.join(mri_root, fdict[pet], 'cat12', fdict[pet] + 'N.nii.gz')
			if pet + '_SUVR.nii' in os.listdir(os.path.join(pet_root, pet, 'pet_proc')):
				src_pet = os.path.join(pet_root, pet, 'pet_proc', pet + '_SUVR.nii')
			else:
				src_pet = os.path.join(pet_root, pet, 'pet_proc', pet + '_SUVR.nii.gz')
				pet_gz = True

			dest_mri = os.path.join(dest_root, 'images')
			dest_pet = os.path.join(dest_root, 'labels')

			f.write(fdict[pet] + 'N.nii,' + pet + '_SUVR.nii\n')
			# shutil.copy(src_mri, dest_mri)
			shutil.copy(src_pet, os.path.join(dest_pet, fdict[pet] + 'N.nii.gz' if pet_gz else fdict[pet] + 'N.nii'))

def rename_files(src, flist):
	for pid in flist:
		for fname in os.listdir(src):
			if fname.startswith(pid):
				shutil.move(os.path.join(src, fname), os.path.join(src, pid + '.nii.gz'))

def rename_wmparc(src, flist):
	for pid in flist:
		for fname in os.listdir(src):
			if fname.startswith(pid):
				shutil.move(os.path.join(src, fname), os.path.join(src, pid + '.mgz'))

if __name__ == '__main__':

	# move_wmparc()

	mri_list = []
	pet_list = []
	with open('/scratch/ychen855/Data/mri2pet/adni_ptau/data_split_ptau.csv') as f:
		f.readline()
		for line in f:
			linelist = line.strip().split(',')
			mri_id = linelist[0]
			pet_id = linelist[1]
			mri_list.append(mri_id)
			pet_list.append(pet_id)

	# prepare_adni(pet_list)

	# check_dims('/scratch/ychen855/Data/mri2pet/adni/labels', 'label_dims.txt')
	rename_files('/scratch/ychen855/Data/mri2pet/adni_ptau/images', mri_list)
	rename_files('/scratch/ychen855/Data/mri2pet/adni_ptau/labels', pet_list)
	rename_wmparc('/scratch/ychen855/Data/mri2pet/adni_ptau/wmparc_256', mri_list)


	# pet_list = []
	# id_dict = dict()
	# with open('/data/hohokam/Yanxi/Data/mri2pet/adni/id_map.csv') as f:
	# 	for line in f:
	# 		linelist = line[:-1].split(',') if line[-1] == '\n' else line.split(',')
	# 		id_dict[linelist[1].split('_')[0]] = linelist[0]

	# with open('/scratch/ychen855/clinical_adni.csv') as f:
	# 	with open('/data/hohokam/Yanxi/Data/mri2pet/adni/clinical_adni.csv', 'w') as fout:
	# 		fout.write('image,' + f.readline())
	# 		for line in f:
	# 			linelist = line[:-1].split(',') if line[-1] == '\n' else line.split(',')
	# 			pup = linelist[-2]
	# 			fout.write(id_dict[pup] + ',' + line)

	# normalize_img('/scratch/ychen855/Data/mri2pet/674/warped/raw/test/labels', '/scratch/ychen855/Data/mri2pet/674/warped/normalized/test/labels', False)
	# prepare_image()
	# split_train_val('/scratch/ychen855/Data/mri2pet/200/warped', 160, 20, 20)
	# move_fs()

	# flist = []
	# with open('/data/hohokam/Yanxi/Data/mri2pet/adcn/clinical.csv') as f:
	# 	f.readline()
	# 	for line in f:
	# 		linelist = line[:-1].split(',') if line[-1] == '\n' else line.split(',')
	# 		flist.append('w_' + linelist[0])
	# prepare_image(flist, True)
