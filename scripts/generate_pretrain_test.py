import os
import shutil

dataset = 'adni_all'

best = True
continue_train = False

n_epoch = 200
save_freq = 20
multi_gpu = True

lr_policy = 'lambda'
lr = 0.0002
pool_size = 0
lmd = 10.0
lmdidt = 0.3
init_type = 'kaiming'
batch_size = 16

n_iter = 1
n_iter_decay = n_epoch - n_iter

model = 'cycle_gan'
netg = 'resnet_6blocks'
python_path = '/scratch/ychen855/mri2pet/CycleGAN_3D/pretrain_cyclegan.py'

n_fold = 5
n_workers = 4
netd = 'defined' # basic/n_layers/pixel
ngf = 64
ndf = 64

data_root = '/scratch/ychen855/Data/mri2pet'
checkpoints_dir = '/scratch/ychen855/mri2pet/CycleGAN_3D/checkpoints_pretrain'

exp_name = 'cyclegan_pretrain'
data_folder = os.path.join(data_root, dataset)
job_file = 'sbatch_jobs/' + exp_name + '.sh'
with open(job_file, 'w') as f:
    lines = list()
    lines.append('#!/bin/bash\n')
    lines.append('#SBATCH -n 4                        # number of cores')
    lines.append('#SBATCH --mem=128G')
    if not continue_train:
        lines.append('#SBATCH -p general')
    lines.append('#SBATCH -G a100:1')
    if not continue_train:
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

    lines.append('NAME=' + dataset + '_' + exp_name + '_' + str(n_epoch) + 'epoch')
    lines.append('DATA_PATH=' + os.path.join(data_root, dataset))
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
    lines.append('PYTHON_PATH=' + python_path)
    lines.append('CHECKPOINTS_DIR=' + checkpoints_dir)
    lines.append('SAVE_FREQ=' + str(save_freq))
    lines.append('LR_POLICY=' + lr_policy)
    lines.append('N_ITER=' + str(n_iter))
    lines.append('N_ITER_DECAY=' + str(n_iter_decay))

    command = 'python ${PYTHON_PATH} --lr_policy ${LR_POLICY} --init_type ${INIT_TYPE} --lambda_A ${LAMBDA} --lambda_B ${LAMBDA} --lambda_identity ${LAMBDA_IDT} --pool_size ${POOL_SIZE} --name ${NAME} --lr ${LR} --data_path ${DATA_PATH} --model ${MODEL} --workers ${NWORKERS} --batch_size ${BATCH_SIZE} --save_epoch_freq ${SAVE_FREQ} --niter ${N_ITER} --niter_decay ${N_ITER_DECAY} --checkpoints_dir ${CHECKPOINTS_DIR} --netG ${NETG} --netD ${NETD} --ngf ${NGF} --ndf ${NDF}'

    if multi_gpu:
        command += ' --gpu_ids 0,1'

    if continue_train:
        command += ' --continue_train --which_epoch ${WHICH_EPOCH} --epoch_count ${EPOCH_COUNT}'
        
    # if cross_validation:
    #     command += ' --cross_validation'
    lines.append(command + '\n')

    f.write('\n'.join(lines))
