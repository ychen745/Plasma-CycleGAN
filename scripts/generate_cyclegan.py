import os
import shutil

splits = [0]
bbbm = 'ptau'
dataset = 'adni_' + bbbm

best = True
training = True
use_pretrain = False
pretrain_weights = '/scratch/ychen855/mri2pet/CycleGAN_3D/checkpoints_' + bbbm
continue_train = False

parallel = True

n_epoch = 80
save_freq = 20
which_epochs = [80]
n_iter = 1
n_iter_decay = n_epoch - n_iter

cross_validation = True

lr_policy = 'lambda'
lr = 0.0002
pool_size = 0
lmd = 10.0
lmdidt = 0.3
init_type = 'kaiming'
batch_size = 8

model = 'cycle_gan' if training else 'test'
netg = 'resnet_6blocks'
if training:
    python_path = '/scratch/ychen855/mri2pet/CycleGAN_3D/train_cyclegan.py'
else:
    python_path = '/scratch/ychen855/mri2pet/CycleGAN_3D/test_cyclegan.py'
n_fold = 5
n_workers = 4 if training else 1
netd = 'defined' # basic/n_layers/pixel
ngf = 64
ndf = 64

data_root = '/scratch/ychen855/Data/mri2pet'
gen_root = '/scratch/ychen855/Data/mri2pet/adni_ptau/gen'
checkpoints_dir = '/scratch/ychen855/mri2pet/CycleGAN_3D/checkpoints_' + bbbm

data_split = 'data_split_ptau.csv'

for which_epoch in which_epochs:
    epoch_count = which_epoch
    test_epoch = which_epoch
    for split in splits:
        for i in range(1):
            exp_name = 'cyclegan_ptau'
            data_folder = os.path.join(data_root, dataset)
            if training:
                job_file = 'sbatch_jobs/' + dataset + '_' + exp_name + '.sh'
            elif best:
                job_file = 'sbatch_jobs/' + dataset + '_' + exp_name + '_test_best.sh'
            else:
                job_file = 'sbatch_jobs/' + dataset + '_' + exp_name + '_test_' + str(which_epoch) + '.sh'
            with open(job_file, 'w') as f:
                lines = list()
                lines.append('#!/bin/bash\n')
                if training:
                    lines.append('#SBATCH -n 4                        # number of cores')
                else:
                    lines.append('#SBATCH -n 1                        # number of cores')
                lines.append('#SBATCH --mem=128G')
                if training and not continue_train:
                    lines.append('#SBATCH -p general')
                lines.append('#SBATCH -G a100:1')
                if training and not continue_train:
                    lines.append('#SBATCH -t 1-00:00:00                 # wall time (D-HH:MM:SS)')
                else:
                    lines.append('#SBATCH -t 0-04:00:00                 # wall time (D-HH:MM:SS)')
                lines.append('#SBATCH -o /scratch/ychen855/mri2pet/CycleGAN_3D/scripts/job_logs/' + exp_name + '.out')
                lines.append('#SBATCH -e /scratch/ychen855/mri2pet/CycleGAN_3D/scripts/job_logs/' + exp_name + '.err')
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

                    command = 'python ${PYTHON_PATH} --lr_policy ${LR_POLICY} --init_type ${INIT_TYPE} --lambda_A ${LAMBDA} --lambda_B ${LAMBDA} --lambda_identity ${LAMBDA_IDT} --pool_size ${POOL_SIZE} --name ${NAME} --lr ${LR} --data_path ${DATA_PATH} --val_path ${VAL_PATH} --model ${MODEL} --workers ${NWORKERS} --batch_size ${BATCH_SIZE} --save_epoch_freq ${SAVE_FREQ} --niter ${N_ITER} --niter_decay ${N_ITER_DECAY} --checkpoints_dir ${CHECKPOINTS_DIR} --netG ${NETG} --netD ${NETD} --ngf ${NGF} --ndf ${NDF} --data_split ${DATA_SPLIT} --fold ${FOLD}'

                    if parallel:
                        command += ' --gpu_ids 0,1'

                    if continue_train:
                        lines.append('WHICH_EPOCH=' + str(which_epoch))
                        lines.append('EPOCH_COUNT=' + str(epoch_count))
                        command += ' --continue_train --which_epoch ${WHICH_EPOCH} --epoch_count ${EPOCH_COUNT}'
                    if cross_validation:
                        command += ' --cross_validation'
                    lines.append(command + '\n')
                else:
                    lines.append('DATA_PATH=' + os.path.join(data_root, dataset))
                    lines.append('MODEL=' + model)
                    lines.append('NETG=' + netg)
                    lines.append('WHICH_EPOCH=' + str(test_epoch))
                    lines.append('NAME=' + dataset + '_' + exp_name + '_' + str(n_epoch) + 'epoch')
                    lines.append('FOLD=' + str(i))
                    if best:
                        if exp_name + '_best_' + str(n_epoch) not in os.listdir(os.path.join(gen_root)):
                            os.mkdir(os.path.join(gen_root, exp_name + '_best_' + str(n_epoch)))
                        gen_folder = os.path.join(gen_root, exp_name + '_best_' + str(n_epoch))
                    else:
                        if exp_name + '_' + str(which_epoch) + '_' + str(n_epoch) not in os.listdir(os.path.join(gen_root)):
                            os.mkdir(os.path.join(gen_root, exp_name + '_' + str(which_epoch) + '_' + str(n_epoch)))
                        gen_folder = os.path.join(gen_root, exp_name + '_' + str(which_epoch) + '_' + str(n_epoch))
                    
                    lines.append('RES_DIR=' + gen_folder)
                    lines.append('NWORKERS=' + str(n_workers))
                    lines.append('PYTHON_PATH=' + python_path)
                    lines.append('NGF=' + str(ngf))
                    lines.append('CHECKPOINTS_DIR=' + checkpoints_dir)
                    lines.append('DATA_SPLIT=' + os.path.join(data_root, dataset, data_split))

                    lines.append('IMG_DIR=' + os.path.join(data_root, dataset, 'images'))

                    command = 'python ${PYTHON_PATH} --data_path ${DATA_PATH} --data_split ${DATA_SPLIT} --name ${NAME} --result ${RES_DIR} --workers ${NWORKERS} --netG ${NETG} --model ${MODEL} --ngf ${NGF} --checkpoints_dir ${CHECKPOINTS_DIR} --which_epoch ${WHICH_EPOCH}'
                    if cross_validation:
                        command += ' --cross_validation --fold ${FOLD}'
                    if best:
                        command += ' --best'
                    lines.append(command + '\n')

                f.write('\n'.join(lines))
