import os
import SimpleITK as sitk
import numpy as np
import nibabel as nib

def read_image(path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    image = reader.Execute()
    return image

def resample_size(image, target_size, interpolator=sitk.sitkLinear):

    dimension = image.GetDimension()
    reference_physical_size = np.zeros(image.GetDimension())
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(image.GetSize(), image.GetSpacing(), reference_physical_size)]
    
    reference_origin = image.GetOrigin()
    reference_direction = image.GetDirection()

    reference_size = target_size
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(image.GetDirection())

    transform.SetTranslation(np.array(image.GetOrigin()) - reference_origin)
  
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(image.TransformContinuousIndexToPhysicalPoint(np.array(image.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    # centered_transform = sitk.Transform(transform)
    centered_transform = sitk.CompositeTransform([sitk.Transform(transform), sitk.Transform(centering_transform)])

    # sitk.Show(sitk.Resample(original_CT, reference_image, centered_transform, sitk.sitkLinear, 0.0))
    
    return sitk.Resample(image, reference_image, centered_transform, interpolator, 0.0)

if __name__ == '__main__':
    target_size = (256, 256, 256)
    gen_path = '/data/hohokam/Yanxi/Data/mri2pet/adni/gen'
    exp_names = ['cyclegan_80_100', 'pix2pix_20_20', 'pix2pix_abeta_concat_20_20', 'sharegan_80_80', 'sharegan_concat_120_120']
    # exp_names = ['abeta_concat_60_80']
    for exp_name in exp_names:
        if exp_name + '_256' not in os.listdir(gen_path):
            os.mkdir(os.path.join(gen_path, exp_name + '_256'))
        for pid in os.listdir(os.path.join(gen_path, exp_name)):
            image = read_image(os.path.join(gen_path, exp_name, pid))
            resampled_image = resample_size(image, target_size)
            sitk.WriteImage(resampled_image, os.path.join(gen_path, exp_name + '_256', pid))