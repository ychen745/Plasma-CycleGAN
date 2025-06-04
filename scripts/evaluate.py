import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.stats import ttest_rel, ttest_ind
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import argparse
from scipy.stats import pearsonr
import scipy.ndimage as ndimage


def evaluate(gen_path, true_path, map_dict, log_file, log_ssim):
    ssim = []
    psnr = []
    mse = []

    best_mri = ''
    best_ssim = 0

    for f in os.listdir(gen_path):
        gen_image = nib.load(os.path.join(gen_path, f)).get_fdata()
        # gen_image = np.transpose(gen_image, (1, 2, 0))
        true_image = nib.load(os.path.join(true_path, map_dict[f])).get_fdata()
        np.nan_to_num(gen_image, copy=False, nan=0.0)
        np.nan_to_num(true_image, copy=False, nan=0.0)

        gen_image = (gen_image - np.mean(gen_image)) / np.std(gen_image)
        true_image = (true_image - np.mean(true_image)) / np.std(true_image)
        
        gen_image = gen_image - np.min(gen_image)
        gen_image = gen_image / (np.max(gen_image) - np.min(gen_image)) * 255.0
        true_image = true_image - np.min(true_image)
        true_image = true_image / (np.max(true_image) - np.min(true_image)) * 255.0

        cur_ssim = structural_similarity(true_image, gen_image, data_range=255.0, full=False)
        ssim.append(cur_ssim)
        cur_psnr = peak_signal_noise_ratio(true_image, gen_image, data_range=255.0)
        psnr.append(cur_psnr)
        cur_mse = mean_squared_error(true_image, gen_image)
        mse.append(cur_mse)

        if cur_ssim > best_ssim:
            best_ssim = cur_ssim
            best_mri = f

    ssim_arry = np.array(ssim)
    psnr_arry = np.array(psnr)
    mse_arry = np.array(mse)

    mean_ssim = np.mean(ssim_arry)
    std_ssim = np.std(ssim_arry)
    mean_psnr = np.mean(psnr_arry)
    std_psnr = np.std(psnr_arry)
    mean_mse = np.mean(mse_arry)
    std_mse = np.std(mse_arry)

    with open(log_file, 'w') as f:
        f.write('mean_ssim,std_ssim,mean_psnr,std_psnr,mean_mes,std_mse\n')
        f.write(','.join([str(ele) for ele in [mean_ssim, std_ssim, mean_psnr, std_psnr, mean_mse, std_mse]]) + '\n')

    with open(log_ssim, 'w') as f:
        f.write('best ssim: ' + str(best_ssim) + ' image: ' + best_mri)

    # print('best ssim: ', best_ssim)
    # print('best mri: ', best_mri)
    return (ssim, psnr, mse)

def evaluate_all(gen_paths, true_path, map_dict, log_file):
    ssims = []
    psnrs = []
    mses = []
    for gen_path in gen_paths:
        ssim, psnr, mse = evaluate(gen_path, true_path, map_dict, log_file)
        ssims += ssim
        psnrs += psnr
        mses += mse
    
    ssim_arry = np.array(ssims)
    psnr_arry = np.array(psnrs)
    mse_arry = np.array(mses)

    mean_ssim = np.mean(ssim_arry)
    std_ssim = np.std(ssim_arry)
    mean_psnr = np.mean(psnr_arry)
    std_psnr = np.std(psnr_arry)
    mean_mse = np.mean(mse_arry)
    std_mse = np.std(mse_arry)

    with open(log_file, 'w') as f:
        f.write('mean_ssim,std_ssim,mean_psnr,std_psnr,mean_mes,std_mse\n')
        f.write(','.join([str(ele) for ele in [mean_ssim, std_ssim, mean_psnr, std_psnr, mean_mse, std_mse]]) + '\n')

def rot_wmparc_128(image):
    rotated_image = ndimage.rotate(image, angle=-90, axes=(1, 2), reshape=False)
    rotated_image = np.flip(rotated_image, axis=(0, 1, 2))
    return rotated_image

def rot_wmparc_256(image):
    image = ndimage.rotate(image, angle=180, axes=(0, 1), reshape=False)
    image = rot_wmparc_128(image)
    return image

def get_suvr(petfile, wmparcfile, vqgan=True, res=128):
    ### petfile is the path of pet image need to calculate MCSUVR:str
    ### wmparcfile is the path of standard pet inage: str
    V_pet=nib.load(petfile).get_fdata()
    # if vqgan:
    #     V_pet = np.transpose(V_pet, (1, 2, 0))
    np.nan_to_num(V_pet, copy=False, nan=0.0)

    V_wmparc=nib.load(wmparcfile).get_fdata()
    if res == 128:
        V_wmparc = rot_wmparc_128(V_wmparc)
    elif res == 256:
        V_wmparc = rot_wmparc_256(V_wmparc)
    else:
        print('Invalid resolution.')
        exit()

    CBmask=np.zeros(V_pet.shape)
    CBWmask=np.array(CBmask,copy=True)
    MCmask=np.array(CBmask,copy=True)

    ##Reference Region ---Cerebellum cortex
    CBmask[V_wmparc == 8]=1
    CBmask[V_wmparc == 47]=1

    ##Target Region -- Mean cortical Regions
    MCmask[V_wmparc==1012]=1 #lh-LOF (lateral-orbital-frontal)
    MCmask[V_wmparc==2012]=1 #rh-LOF
    MCmask[V_wmparc==1014]=1 #lh-MOF (medial-orbital-frontal)
    MCmask[V_wmparc==2014]=1 #rh-MOF
    MCmask[V_wmparc==1015]=1 #lh-MT  (middle-temporal)
    MCmask[V_wmparc==2015]=1 #rh-MT
    MCmask[V_wmparc==1030]=1 #lh-ST  (superior-temporal)
    MCmask[V_wmparc==2030]=1 #rh-ST
    MCmask[V_wmparc==1025]=1 #lh-PREC (precuneus)
    MCmask[V_wmparc==2025]=1 #rh-PREC
    MCmask[V_wmparc==1027]=1 #lh-RMF (rostral-middle-frontal)
    MCmask[V_wmparc==2027]=1 #rh-RMF
    MCmask[V_wmparc==1028]=1 #lh-SF  (superior-frontal)
    MCmask[V_wmparc==2028]=1 #rh-SF

    MC_suvr=np.mean(V_pet[MCmask==1])/np.mean(V_pet[CBmask==1])

    return MC_suvr

def evaluate_pixcorr(gen_pet, true_pet):
    gen_image = nib.load(gen_pet).get_fdata()
    true_image = nib.load(true_pet).get_fdata()

    gen_arry = gen_image.flatten()
    true_arry = true_image.flatten()

    r, pval = pearsonr(gen_arry, true_arry)

    return r, pval

def regional_suvr(gen_pet, true_pet, wmparcfile):
    gen_image = nib.load(gen_pet).get_fdata()
    np.nan_to_num(gen_image, copy=False, nan=0.0)

    true_image = nib.load(true_pet).get_fdata()
    np.nan_to_num(true_image, copy=False, nan=0.0)

    wmparc_image = nib.load(wmparcfile).get_fdata()
    wmparc_image = np.round(wmparc_image).astype(int)

    CBmask = np.zeros(wmparc_image.shape)

    CBmask[wmparc_image == 8]=1
    CBmask[wmparc_image == 47]=1

    region_list = ['lh-LOF', 'rh-LOF', 'lh-MOF', 'rh-MOF', 'lh-MT', 'rh-MT', 'lh-ST', 'rh-ST', 'lh-PREC', 'rh-PREC', 'lh-RMF', 'rh-RMF', 'lh-SF', 'rh-SF']
    value_list = [1012, 2012, 1014, 2014, 1015, 2015, 1030, 2030, 1025, 2025, 1027, 2027, 1028, 2028]

    gen_list = []
    true_list = []

    for i in range(len(region_list)):
        gen_list.append(np.mean(gen_image[wmparc_image==value_list[i]]) / np.mean(gen_image[CBmask==1]))
        true_list.append(np.mean(true_image[wmparc_image==value_list[i]]) / np.mean(true_image[CBmask==1]))

    return gen_list, true_list
    

if __name__ == '__main__':
    ### evaluate performance
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--gen_root', type=str)
    parser.add_argument('--data_split', type=str)
    parser.add_argument('--result', type=str)
    parser.add_argument('--ssim', action='store_true')
    parser.add_argument('--suvr', action='store_true')
    parser.add_argument('--pixcorr', action='store_true')
    parser.add_argument('--regional_suvr', action='store_true')
    parser.add_argument('--regional_pixcorr', action='store_true')
    parser.add_argument('--log_suvr', type=str)
    parser.add_argument('--log_pixcorr', type=str)
    parser.add_argument('--log_regional_suvr', type=str)
    parser.add_argument('--log_regional_pixcorr', type=str)
    parser.add_argument('--log_ssim', type=str)
    parser.add_argument('--wmparc', type=str)
    parser.add_argument('--best', action='store_true')


    args = parser.parse_args()
    true_path = os.path.join(args.data_root, args.dataset, 'labels')
    gen_path = os.path.join(args.gen_root, args.exp)

    map_dict = {}

    with open(args.data_split) as f:
        for line in f:
            linelist = line.split('\n')[0].split(',')
            map_dict[linelist[0] + '.nii'] = linelist[1] + '.nii'

    if args.ssim:
        print('ssim')
        log_file = os.path.join(args.result, 'log_' + args.exp + '.txt')
        evaluate(gen_path, true_path, map_dict, log_file, os.path.join(args.log_ssim, args.exp + '.txt'))
        
    if args.suvr:
        print('suvr')
        log_suvr = os.path.join(args.log_suvr, args.exp + '.txt')
        with open(log_suvr, 'w') as f:
            f.write('mri,pet,gen_suvr,true_suvr\n')
            for pid in os.listdir(gen_path):
                pet_id = map_dict[pid]
                gen_suvr = get_suvr(os.path.join(gen_path, pid), os.path.join(args.wmparc, pid), vqgan=True)
                true_suvr = get_suvr(os.path.join(true_path, map_dict[pid]), os.path.join(args.wmparc, pid), vqgan=False)
            
                f.write(','.join([pid, pet_id, str(gen_suvr), str(true_suvr)]) + '\n')
    
    # if args.pixcorr:
    #     print('pixcorr')
    #     rs = []
    #     p_vals = []
    #     log_pixcorr = os.path.join(args.log_pixcorr, args.exp + '.txt')
    #     with open(log_pixcorr, 'w') as f:
    #         f.write('mri,pet,r,p-val\n')
    #         for pid in os.listdir(gen_path):
    #             r, p_val = evaluate_pixcorr(os.path.join(gen_path, pid), os.path.join(true_path, map_dict[pid]))
    #             rs.append(r)
    #             p_vals.append(p_val)
    #             f.write(','.join([pid, map_dict[pid], str(r), str(p_val)]) + '\n')
    #         p_vals = np.array(p_vals)

    #         f.write('avg p-val: ' + str(np.mean(p_vals)) + '\n')
    #         f.write('max p-val: ' + str(np.max(p_vals)) + '\n')
    #         f.write('min p-val: ' + str(np.min(p_vals)) + '\n')
    #         f.write('number of significant pix_corr: ' + str(len(np.where(p_vals < 1e-2)[0])) + '\n')
        
    #     print('avg p-val: ', np.mean(p_vals))
    #     print('max p-val: ', np.mean(p_vals))
    #     print('min p-val: ', np.mean(p_vals))
    #     print('number of significant pix_corr: ', len(np.where(p_vals < 1e-2)[0]))
    
    # if args.regional_suvr:
    #     print('regional suvr')
    #     region_list = ['lh-LOF', 'rh-LOF', 'lh-MOF', 'rh-MOF', 'lh-MT', 'rh-MT', 'lh-ST', 'rh-ST', 'lh-PREC', 'rh-PREC', 'lh-RMF', 'rh-RMF', 'lh-SF', 'rh-SF']
    #     log_regional_suvr = os.path.join(args.log_regional_suvr, args.exp + '_' + str(args.which_epoch) + '_' + str(args.n_epoch) + '.txt')
    #     with open(log_regional_suvr, 'w') as f:
    #         f.write(','.join(['mri', 'pet'] + [ele + '_gen,' + ele + '_true' for ele in region_list]) + '\n')
    #         for pid in os.listdir(gen_path):
    #             gen_pet = os.path.join(gen_path, pid)
    #             true_pet = os.path.join(true_path, map_dict[pid])
    #             wmparc_file = os.path.join(args.wmparc, pid)
    #             gen_list, true_list = regional_suvr(gen_pet, true_pet, wmparc_file)
    #             f.write(','.join([pid, map_dict[pid]] + [str(gen_list[i]) + ',' + str(true_list[i]) for i in range(len(region_list))]) + '\n')

    # if args.regional_pixcorr:
    #     print('regional pixcorr')
    #     region_list = ['lh-LOF', 'rh-LOF', 'lh-MOF', 'rh-MOF', 'lh-MT', 'rh-MT', 'lh-ST', 'rh-ST', 'lh-PREC', 'rh-PREC', 'lh-RMF', 'rh-RMF', 'lh-SF', 'rh-SF']
    #     log_regional_pixcorr = os.path.join(args.log_regional_pixcorr, args.exp + '_' + str(args.which_epoch) + '_' + str(args.n_epoch) + '.txt')
    #     with open(log_regional_pixcorr, 'w') as f:
    #         f.write(','.join(['mri', 'pet'] + [ele + '_r,' + ele + '_p' for ele in region_list]) + '\n')
    #         for pid in os.listdir(gen_path):
    #             gen_pet = os.path.join(gen_path, pid)
    #             true_pet = os.path.join(true_path, map_dict[pid])
    #             wmparc_file = os.path.join(args.wmparc, pid)
    #             r_list, p_list = regional_pixcorr(gen_pet, true_pet, wmparc_file)
    #             f.write(','.join([pid, map_dict[pid]] + [str(r_list[i]) + ',' + str(p_list[i]) for i in range(len(region_list))]) + '\n')

