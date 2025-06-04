import os
import shutil

root = '../checkpoints_ptau'
src = '120'
tg = '200'

for check in os.listdir(root):
    if check.endswith(src + 'epoch') and 'mid' in check and 'cyclegan' in check:
        shutil.move(os.path.join(root, check), os.path.join(root, '_'.join(check.split('_')[:-1]) + '_' + tg + 'epoch'))