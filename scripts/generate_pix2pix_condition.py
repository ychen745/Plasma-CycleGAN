import os
import shutil

bbbm = 'ptau'
dataset = 'adni_' + bbbm

training = False
continue_train = True

n_epoch = 120
n_iter = 1
n_iter_decay = n_epoch - n_iter
save_freq = 20
which_epochs = [60, 80, 100, 120]

cross_validation = True

lr_policy = 'lambda'
lr = 0.0002
pool_size = 20
lmd = 10.0
lmdidt = 0.3
init_type = 'kaiming'
batch_size = 16

# condition = True
conditions = ['ptau_lp']
# conditions = ['abeta']

model = 'pix2pix3d_condition' if training else 'pix2pix3d_condition_test'
# netg = 'resnet_6blocks'
netg = 'condition'

if training:
    python_path = '/scratch/ychen855/mri2pet/CycleGAN_3D/train_pix2pix_condition.py'
else:
    python_path = '/scratch/ychen855/mri2pet/CycleGAN_3D/test_pix2pix_condition.py'

n_fold = 5
n_workers = 4 if training else 1
netd = 'defined' # basic/n_layers/pixel

ngf = 64
ndf = 64

data_root = '/scratch/ychen855/Data/mri2pet'
gen_root = '/scratch/ychen855/Data/mri2pet'
checkpoints_dir = '/scratch/ychen855/mri2pet/CycleGAN_3D/checkpoints_' + bbbm

data_split = 'data_split_ptau_lp_mid.csv'

# bbbm_modes = ['image', 'add', 'concat']
bbbm_modes = ['concat']

for which_epoch in which_epochs:
    epoch_count = which_epoch
    test_epoch = which_epoch
    for condition in conditions:
        for bbbm_mode in bbbm_modes:
            for i in range(1):
                exp_name = 'pix2pix_c_' + condition + '_mid'
                data_folder = os.path.join(data_root, dataset)
                if training:
                    job_file = 'sbatch_jobs/' + dataset + '_' + exp_name + '_' + str(n_epoch) + '.sh'
                else:
                    job_file = 'sbatch_jobs/' + dataset + '_' + exp_name + '_test_' + str(which_epoch) + '_' + str(n_epoch) + 'epoch.sh'
                with open(job_file, 'w') as f:
                    lines = list()
                    lines.append('#!/bin/bash\n')
                    if training:
                        lines.append('#SBATCH -n 4                        # number of cores')
                    else:
                        lines.append('#SBATCH -n 1                        # number of cores')
                    lines.append('#SBATCH --mem=128G')
                    # if training:
                    #     lines.append('#SBATCH -p general')
                    lines.append('#SBATCH -G a100:1')
                    if training:
                        lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
                    else:
                        lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
                    lines.append('#SBATCH -o /scratch/ychen855/mri2pet/CycleGAN_3D/scripts_yc/job_logs/' + exp_name + '.out             # STDOUT (%j = JobId)')
                    lines.append('#SBATCH -e /scratch/ychen855/mri2pet/CycleGAN_3D/scripts_yc/job_logs/' + exp_name + '.err             # STDERR (%j = JobId)')
                    lines.append('#SBATCH --mail-type=END             # Send a notification when the job starts, stops, or fails')
                    lines.append('#SBATCH --mail-user=ychen855@asu.edu # send-to address\n')

                    lines.append('module purge    # Always purge modules to ensure a consistent environment\n')

                    lines.append('module load cuda-12.4.1-gcc-12.1.0\n')

                    lines.append('source ~/.bashrc')
                    lines.append('conda activate cu124\n')
                    if training:
                        lines.append('NAME=' + dataset + '_' + exp_name + '_' + str(n_epoch) + 'epoch')
                        lines.append('FOLD=' + str(i))
                        lines.append('DATA_PATH=' + os.path.join(data_root, dataset))
                        lines.append('VAL_PATH=' + os.path.join(data_root, dataset))
                        lines.append('MODEL=' + model)
                        lines.append('NETG=' + netg)
                        lines.append('NETD=' + netd)
                        lines.append('NGF=' + str(ngf))
                        lines.append('NDF=' + str(ndf))
                        lines.append('LR=' + str(lr))
                        lines.append('LAMBDA=' + str(lmd))
                        lines.append('LAMBDA_IDT=' + str(lmdidt))
                        lines.append('INIT_TYPE=' + str(init_type))
                        lines.append('POOL_SIZE=' + str(pool_size))
                        lines.append('NWORKERS=' + str(n_workers))
                        lines.append('BATCH_SIZE=' + str(batch_size))
                        lines.append('DATA_SPLIT=' + os.path.join(data_root, dataset, data_split))
                        lines.append('PYTHON_PATH=' + python_path)
                        lines.append('CHECKPOINTS_DIR=' + checkpoints_dir)
                        lines.append('SAVE_FREQ=' + str(save_freq))
                        lines.append('LR_POLICY=' + lr_policy)
                        lines.append('N_ITER=' + str(n_iter))
                        lines.append('N_ITER_DECAY=' + str(n_iter_decay))
                        lines.append('BBBM_MODE=' + bbbm_mode)

                        command = 'python ${PYTHON_PATH} --lr_policy ${LR_POLICY} --init_type ${INIT_TYPE} --pool_size ${POOL_SIZE} --name ${NAME} --lr ${LR} --lambda_A ${LAMBDA} --lambda_B ${LAMBDA} --lambda_identity ${LAMBDA_IDT} --data_path ${DATA_PATH} --val_path ${VAL_PATH} --model ${MODEL} --workers ${NWORKERS} --batch_size ${BATCH_SIZE} --save_epoch_freq ${SAVE_FREQ} --niter ${N_ITER} --niter_decay ${N_ITER_DECAY} --checkpoints_dir ${CHECKPOINTS_DIR} --netG ${NETG} --ngf ${NGF} --ndf ${NDF} --data_split ${DATA_SPLIT} --fold ${FOLD} --bbbm_mode ${BBBM_MODE}'
                        if cross_validation:
                            command += ' --cross_validation'

                        lines.append('CONDITION=' + condition)
                        command += ' --condition ${CONDITION}'
                        if continue_train:
                            lines.append('WHICH_EPOCH=' + str(which_epoch))
                            lines.append('EPOCH_COUNT=' + str(epoch_count))
                            command += ' --continue_train --which_epoch ${WHICH_EPOCH} --epoch_count ${EPOCH_COUNT}'
                        lines.append(command + '\n')
                    else:
                        lines.append('DATA_PATH=' + os.path.join(data_root, dataset))
                        lines.append('MODEL=' + model)
                        lines.append('NETG=' + netg)
                        lines.append('FOLD=' + str(i))
                        lines.append('WHICH_EPOCH=' + str(test_epoch))
                        lines.append('NAME=' + dataset + '_' + exp_name + '_' + str(n_epoch) + 'epoch')
                        if exp_name + '_' + str(which_epoch) + '_' + str(n_epoch) not in os.listdir(os.path.join(gen_root, dataset, 'gen')):
                            os.mkdir(os.path.join(gen_root, dataset, 'gen', exp_name + '_' + str(which_epoch) + '_' + str(n_epoch)))
                        gen_folder = os.path.join(gen_root, dataset, 'gen', exp_name + '_' + str(which_epoch) + '_' + str(n_epoch))
                        
                        lines.append('RES_DIR=' + gen_folder)
                        lines.append('NWORKERS=1')
                        lines.append('PYTHON_PATH=' + python_path)
                        lines.append('NGF=' + str(ngf))
                        lines.append('CHECKPOINTS_DIR=' + checkpoints_dir)
                        lines.append('DATA_SPLIT=' + os.path.join(data_root, dataset, data_split))
                        lines.append('IMG_DIR=' + os.path.join(data_root, dataset, 'images'))
                        lines.append('BBBM_MODE=' + bbbm_mode)

                        command = 'python ${PYTHON_PATH} --data_path ${DATA_PATH} --data_split ${DATA_SPLIT} --name ${NAME} --result ${RES_DIR} --workers ${NWORKERS} --netG ${NETG} --model ${MODEL} --ngf ${NGF} --checkpoints_dir ${CHECKPOINTS_DIR} --which_epoch ${WHICH_EPOCH} --bbbm_mode ${BBBM_MODE}'
                        
                        if cross_validation:
                            command += ' --cross_validation --fold ${FOLD}'
                        lines.append('CONDITION=' + condition)
                        command += ' --condition ${CONDITION}'
                        lines.append(command)

                    f.write('\n'.join(lines))
