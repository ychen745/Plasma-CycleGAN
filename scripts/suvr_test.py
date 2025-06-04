import os
import numpy as np
import nibabel as nib

def get_suvr(petfile, wmparcfile):
    ### petfile is the path of pet image need to calculate MCSUVR:str
    ### wmparcfile is the path of standard pet inage: str
    V_pet=nib.load(petfile).get_fdata()
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

if __name__ == '__main__':
    pet_file = '/data/hohokam/Yanxi/Data/mri2pet/adni/labels_256/FBP012S4545L031716N4.nii'
    wmparc_file = '/data/hohokam/Yanxi/Data/mri2pet/adni/wmparc/m012S4545L031716N63TCF/wmparc.mgz'

    mcsuvr = get_suvr(pet_file, wmparc_file)
    print(mcsuvr)

    # 0.678
    # 0.6855