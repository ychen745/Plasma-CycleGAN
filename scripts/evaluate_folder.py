import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib
from scipy.stats import ttest_rel, ttest_ind
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import argparse

def evaluate(gen_path, true_path, map_dict, log_file):
    ssim = []
    psnr = []
    mse = []
    for f in os.listdir(gen_path):
        gen_image = nib.load(os.path.join(gen_path, f)).get_fdata()
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

    return (ssim, psnr, mse)

def evaluate_all(gen_paths, true_path, fmap, log_file):

    map_dict = {}
    with open(fmap) as f:
        for line in f:
            linelist = line.split('\n')[0].split(',')
            map_dict[linelist[0]] = linelist[1]

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

def get_suvr(petfile, wmparcfile):
    ### petfile is the path of pet image need to calculate MCSUVR:str
    ### wmparcfile is the path of standard pet inage: str
    V_pet=nib.load(petfile).get_fdata().squeeze()
    np.nan_to_num(V_pet, copy=False, nan=0.0)

    V_wmparc=nib.load(wmparcfile).get_fdata()
    V_wmparc=np.round(V_wmparc).astype(int)

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

def evaluate_suvr(gen_pet, true_pet, wmparc):
    suvr_gen = get_suvr(gen_pet, wmparc)
    suvr_true = get_suvr(true_pet, wmparc)
    if (suvr_gen - 1.1) * (suvr_true - 1.1) > 0:
        return 1
    else:
        return 0

def evaluate_folder(gen_path, true_path, wmparc_path, log_path):
    ssim = []
    psnr = []
    mse = []
    suvr = []
    for f in os.listdir(gen_path):
        gen_image = nib.load(os.path.join(gen_path, f)).get_fdata().squeeze()
        true_image = nib.load(os.path.join(true_path, f)).get_fdata().squeeze()
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

        gen_suvr = get_suvr(os.path.join(gen_path, f), os.path.join(wmparc_path, f))
        true_suvr = get_suvr(os.path.join(true_path, f), os.path.join(wmparc_path, f))

        suvr.append(gen_suvr - true_suvr)

    ssim_arry = np.array(ssim)
    psnr_arry = np.array(psnr)
    mse_arry = np.array(mse)
    suvr_arry = np.array(suvr)

    mean_ssim = np.mean(ssim_arry)
    std_ssim = np.std(ssim_arry)
    mean_psnr = np.mean(psnr_arry)
    std_psnr = np.std(psnr_arry)
    mean_mse = np.mean(mse_arry)
    std_mse = np.std(mse_arry)
    mean_suvr = np.mean(suvr_arry)
    std_suvr = np.std(suvr_arry)

    with open(log_path, 'w') as f:
        f.write('mean_ssim,std_ssim,mean_psnr,std_psnr,mean_mes,std_mse,mean_suvr_gap,std_suvr_gap\n')
        f.write(','.join([str(ele) for ele in [mean_ssim, std_ssim, mean_psnr, std_psnr, mean_mse, std_mse, mean_suvr, std_suvr]]) + '\n')

if __name__ == '__main__':
    ### evaluate performance

    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str)
    parser.add_argument('--true_path', type=str)
    parser.add_argument('--wmparc_path', type=str)
    parser.add_argument('--log_path', type=str)

    args = parser.parse_args()
    
    evaluate_folder(args.gen_path, args.true_path, args.wmparc_path, args.log_path)
        
    # if args.suvr:
    #     n = 0
    #     correct = 0
    #     for pid in os.listdir(gen_path):
    #         n += 1
    #         correct += evaluate_suvr(os.path.join(gen_path, pid), os.path.join(true_path, map_dict[pid]), os.path.join(args.wmparc, pid))
    #     with open(log_file, 'a') as f:
    #         f.write('SUVR accuracy: ' + str(correct / n) + '\n')