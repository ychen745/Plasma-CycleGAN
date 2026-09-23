import os
import shutil

bbbm = 'ptau'

data_root = '/scratch/ychen855/Data/mri2pet'
dataset = 'adni_' + bbbm
# gen_root = '/scratch/ychen855/Data/mri2pet/adni_' + bbbm + '/gen'
gen_root = '/scratch/ychen855/Data/mri2pet/adni_ptau/gen'
wmparc = '/scratch/ychen855/Data/mri2pet/adni_' + bbbm + '/wmparc'

result = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/log_ptau'
log_suvr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/suvr_ptau'
log_ssim = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/ssim_ptau'

log_pixcorr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/pixcorr'
log_regional_suvr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/regional_suvr'
log_regional_pixcorr = '/scratch/ychen855/mri2pet/CycleGAN_3D/results/regional_pixcorr'

ssim = True
suvr = True
regional_suvr = False
pixcorr = False
regional_pixcorr = False

# exp_names = ['mri2mri_128_4_19', 'mri2mri_128_4_39', 'mri2mri_128_4_59', 'mri2mri_128_4_79', 'pet2pet_128_4_19', 'pet2pet_128_4_39', 'pet2pet_128_4_59', 'pet2pet_128_4_79']
exp_names = ['cyclegan_pe_60_200', 'cyclegan_pe_80_200', 'cyclegan_pe_100_200', 'cyclegan_pe_120_200',
            'cyclegan_pe_140_200', 'cyclegan_pe_160_200', 'cyclegan_pe_180_200', 'cyclegan_pe_200_200',
            'cyclegan_ptau_60_200', 'cyclegan_ptau_80_200', 'cyclegan_ptau_100_200', 'cyclegan_ptau_120_200',
            'cyclegan_ptau_140_200', 'cyclegan_ptau_160_200', 'cyclegan_ptau_180_200', 'cyclegan_ptau_200_200',
            'cyclegan_ptau_c2n_60_200', 'cyclegan_ptau_c2n_80_200', 'cyclegan_ptau_c2n_100_200', 'cyclegan_ptau_c2n_120_200',
            'cyclegan_ptau_c2n_140_200', 'cyclegan_ptau_c2n_160_200', 'cyclegan_ptau_c2n_180_200', 'cyclegan_ptau_c2n_200_200',
            'pe_ptau_c2n_60_200', 'pe_ptau_c2n_80_200', 'pe_ptau_c2n_100_200', 'pe_ptau_c2n_120_200',
            'pe_ptau_c2n_140_200', 'pe_ptau_c2n_160_200', 'pe_ptau_c2n_180_200', 'pe_ptau_c2n_200_200']
# exp_names = ['256_4_src0.0001_dst0.0001_code512_ptau_c2n']

data_split = os.path.join(data_root, dataset, 'data_split_ptau.csv')

for exp_name in exp_names:
    job_file = os.path.join('sbatch_eval', 'eval_' + exp_name + '.sh')
    with open(job_file, 'w') as f:
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#SBATCH -n 4                        # number of cores')
        lines.append('#SBATCH -N 1                        # number of nodes')
        lines.append('#SBATCH --mem=64G')
        lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
        lines.append('#SBATCH -o /scratch/ychen855/mri2pet/CycleGAN_3D/scripts/log_eval/' + exp_name + '.out')
        lines.append('#SBATCH -e /scratch/ychen855/mri2pet/CycleGAN_3D/scripts/log_eval/' + exp_name + '.err')
        lines.append('#SBATCH --mail-type=NONE             # Send a notification when the job starts, stops, or fails')
        lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

        lines.append('source ~/.bashrc')
        lines.append('conda activate cu124\n')

        lines.append('DATA_ROOT=' + data_root)
        lines.append('DATASET=' + dataset)
        lines.append('GEN_ROOT=' + gen_root)
        lines.append('DATA_SPLIT=' + data_split)
        lines.append('EXP=' + exp_name)
        lines.append('RESULT=' + result)
        lines.append('WMPARC=' + wmparc)
        lines.append('LOG_SUVR=' + log_suvr)
        lines.append('LOG_PIXCORR=' + log_pixcorr)
        lines.append('LOG_REGIONAL_SUVR=' + log_regional_suvr)
        lines.append('LOG_REGIONAL_PIXCORR=' + log_regional_pixcorr)
        lines.append('LOG_SSIM=' + log_ssim)
        
        command = 'python /scratch/ychen855/mri2pet/CycleGAN_3D/scripts/evaluate.py --data_root ${DATA_ROOT} --dataset ${DATASET} --gen_root ${GEN_ROOT} --exp ${EXP} --data_split ${DATA_SPLIT} --result ${RESULT}'

        if ssim:
            command += ' --ssim'
            command += ' --log_ssim ${LOG_SSIM}'
        if suvr:
            command += ' --suvr --wmparc ${WMPARC} --log_suvr ${LOG_SUVR}'
        if pixcorr:
            command += ' --pixcorr --log_pixcorr ${LOG_PIXCORR}'
        if regional_suvr:
            command += ' --regional_suvr --wmparc ${WMPARC} --log_regional_suvr ${LOG_REGIONAL_SUVR}'
        if regional_pixcorr:
            command += ' --regional_pixcorr --wmparc ${WMPARC} --log_regional_pixcorr ${LOG_REGIONAL_PIXCORR}'

        lines.append(command + '\n')

        f.write('\n'.join(lines))
