import os
from options.test_options import TestOptions
import sys
from utils.ConditionDataset import *
import utils.ConditionDataset as ConditionDataset_testing
from torch.utils.data import DataLoader
from models import create_model
import math
from torch.autograd import Variable
from tqdm import tqdm
import datetime
import pandas as pd

def from_numpy_to_itk(image_np, image_itk):
    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    image.SetOrigin(image_itk.GetOrigin())
    image.SetDirection(image_itk.GetDirection())
    image.SetSpacing(image_itk.GetSpacing())
    return image


def prepare_batch(image, ijk_patch_indices):
    image_batches = []
    for batch in ijk_patch_indices:
        image_batch = []
        for patch in batch:
            image_patch = image[patch[0]:patch[1], patch[2]:patch[3], patch[4]:patch[5]]
            image_batch.append(image_patch)

        image_batch = np.asarray(image_batch)
        # image_batch = image_batch[:, :, :, :, np.newaxis]
        image_batches.append(image_batch)

    return image_batches


# inference single image
def inference_single(model, image_path, result_path, resample, resolution, patch_size_x,
              patch_size_y, patch_size_z, stride_inplane, stride_layer, condition, batch_size=1):

    # create transformations to image and labels
    transforms1 = [
        ConditionDataset_testing.Resample(resolution, resample)
    ]

    transforms2 = [
        ConditionDataset_testing.Padding((patch_size_x, patch_size_y, patch_size_z))
    ]

    # read image file
    reader = sitk.ImageFileReader()
    reader.SetFileName(image_path)
    image = reader.Execute()

    # normalize the image
    image = Normalization(image)

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    image = castImageFilter.Execute(image)

    # create empty label in pair with transformed image
    label_tfm = sitk.Image(image.GetSize(), sitk.sitkFloat32)
    label_tfm.SetOrigin(image.GetOrigin())
    label_tfm.SetDirection(image.GetDirection())
    label_tfm.SetSpacing(image.GetSpacing())

    sample = {'image': image, 'label': label_tfm}

    for transform in transforms1:
        sample = transform(sample)

    # keeping track on how much padding will be performed before the inference
    image_array = sitk.GetArrayFromImage(sample['image'])
    pad_x = patch_size_x - (patch_size_x - image_array.shape[2])
    pad_y = patch_size_x - (patch_size_y - image_array.shape[1])
    pad_z = patch_size_z - (patch_size_z - image_array.shape[0])

    image_pre_pad = sample['image']

    for transform in transforms2:
        sample = transform(sample)

    image_tfm, label_tfm = sample['image'], sample['label']

    # convert image to numpy array
    image_np = sitk.GetArrayFromImage(image_tfm)
    label_np = sitk.GetArrayFromImage(label_tfm)

    label_np = np.asarray(label_np, np.float32)

    # unify numpy and sitk orientation
    image_np = np.transpose(image_np, (2, 1, 0))
    label_np = np.transpose(label_np, (2, 1, 0))

    # ----------------- Padding the image if the z dimension still is not even ----------------------

    if (image_np.shape[2] % 2) == 0:
        Padding = False
    else:
        image_np = np.pad(image_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        label_np = np.pad(label_np, ((0, 0), (0, 0), (0, 1)), 'edge')
        Padding = True

    # ------------------------------------------------------------------------------------------------

    # a weighting matrix will be used for averaging the overlapped region
    weight_np = np.zeros(label_np.shape)

    # prepare image batch indices
    inum = int(math.ceil((image_np.shape[0] - patch_size_x) / float(stride_inplane))) + 1
    jnum = int(math.ceil((image_np.shape[1] - patch_size_y) / float(stride_inplane))) + 1
    knum = int(math.ceil((image_np.shape[2] - patch_size_z) / float(stride_layer))) + 1

    patch_total = 0
    ijk_patch_indices = []
    ijk_patch_indicies_tmp = []

    for i in range(inum):
        for j in range(jnum):
            for k in range(knum):
                if patch_total % batch_size == 0:
                    ijk_patch_indicies_tmp = []

                istart = i * stride_inplane
                if istart + patch_size_x > image_np.shape[0]:  # for last patch
                    istart = image_np.shape[0] - patch_size_x
                iend = istart + patch_size_x

                jstart = j * stride_inplane
                if jstart + patch_size_y > image_np.shape[1]:  # for last patch
                    jstart = image_np.shape[1] - patch_size_y
                jend = jstart + patch_size_y

                kstart = k * stride_layer
                if kstart + patch_size_z > image_np.shape[2]:  # for last patch
                    kstart = image_np.shape[2] - patch_size_z
                kend = kstart + patch_size_z

                ijk_patch_indicies_tmp.append([istart, iend, jstart, jend, kstart, kend])

                if patch_total % batch_size == 0:
                    ijk_patch_indices.append(ijk_patch_indicies_tmp)

                patch_total += 1

    image_batches = prepare_batch(image_np, ijk_patch_indices)


    for i in tqdm(range(len(image_batches))):
        image_batch = image_batches[i]

        image_batch = (image_batch - 127.5) / 127.5

        image_batch = torch.from_numpy(image_batch[np.newaxis, :, :, :])
        image_batch = Variable(image_batch.cuda())

        # pred = model(batch)
        condition_tensor = torch.tensor(condition).unsqueeze(0).unsqueeze(0)
        model.set_input(image_batch, condition_tensor)
        model.test()
        pred = model.get_current_visuals()
        pred = pred['fake_B']
        pred = pred.squeeze().data.cpu().numpy()

        pred = (pred * 127.5) + 127.5

        istart = ijk_patch_indices[i][0][0]
        iend = ijk_patch_indices[i][0][1]
        jstart = ijk_patch_indices[i][0][2]
        jend = ijk_patch_indices[i][0][3]
        kstart = ijk_patch_indices[i][0][4]
        kend = ijk_patch_indices[i][0][5]
        label_np[istart:iend, jstart:jend, kstart:kend] += pred[:, :, :]
        weight_np[istart:iend, jstart:jend, kstart:kend] += 1.0

    print("{}: Evaluation complete".format(datetime.datetime.now()))

    # eliminate overlapping region using the weighted value
    label_np = (np.float32(label_np) / np.float32(weight_np) + 0.01)

    # removed the 1 pad on z
    if Padding is True:
        label_np = label_np[:, :, 0:(label_np.shape[2] - 1)]

    # removed all the padding
    label_np = label_np[:pad_x, :pad_y, :pad_z]

    # convert back to sitk space
    label = from_numpy_to_itk(label_np, image_pre_pad)
    # ---------------------------------------------------------------------------------------------

    # save label
    writer = sitk.ImageFileWriter()

    if resample is True:
        print("{}: Resampling label back to original image space...".format(datetime.datetime.now()))
        # label = resample_sitk_image(label, spacing=image.GetSpacing(), interpolator='bspline')   # keep this commented

        label = resize(label, (sitk.GetArrayFromImage(image)).shape[::-1], sitk.sitkLinear)
        label.SetDirection(image.GetDirection())
        label.SetOrigin(image.GetOrigin())
        label.SetSpacing(image.GetSpacing())
    else:
        label = label

    writer.SetFileName(result_path)
    writer.Execute(label)
    print("{}: Save evaluate label at {} success".format(datetime.datetime.now(), result_path))

def inference(model, image_folder, image_list, result_folder, resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, condition_dict, batch_size=1):
    image_path = os.path.join(image_folder, 'images')

    for f in image_list:
        inference_single(model, os.path.join(image_path, f), os.path.join(result_folder, f), resample, resolution, patch_size_x, patch_size_y, patch_size_z, stride_inplane, stride_layer, condition_dict[f], batch_size=1)


if __name__ == '__main__':

    opt = TestOptions().parse()

    model = create_model(opt)
    model.setup(opt)

    img_list = []
    condition_list = []

    with open(opt.data_split) as f:
        header = f.readline().strip().split(',')
        condition_idx = header.index(opt.condition)
        print(condition_idx)
        f.readline()
        for line in f:
            linelist = line.strip().split(',')
            group = int(linelist[-1])
            if group == opt.fold:
                img_list.append(linelist[0] + '.nii')
                condition_list.append(float(linelist[condition_idx]))

    # df = pd.read_csv(opt.data_split)
    # if opt.cross_validation:
    #     img_list = list(df[df['group'] == opt.fold]['mri_id'])
    #     condition_list = list(df[df['group'] == opt.fold][opt.condition])
    # else:
    #     img_list = list(df[df['group'] == 4]['mri_id'])
    #     condition_list = list(df[df['group'] == 4][opt.condition])
    
    print('Number of test images: ', len(img_list))
    
    for i in range(len(img_list)):
        image_path = os.path.join(opt.data_path, 'images', img_list[i])
        result_path = os.path.join(opt.result, img_list[i])
        condition = condition_list[i]
        inference_single(model, image_path, result_path, opt.resample, opt.new_resolution, opt.patch_size[0], opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, condition, 1)

    # condition_dict = dict()
    # image_list = []
    # with open(opt.data_split) as f:
    #     f.readline()
    #     for line in f:
    #         linelist = line[:-1].split(',') if line[-1] == '\n' else line.split(',')
    #         condition_dict[linelist[0]] = float(linelist[-2])
    #         image = linelist[0]
    #         group = linelist[-1]
    #         if group == '5':
    #             image_list.append(image)

    # inference(model, opt.image, image_list, opt.result, opt.resample, opt.new_resolution, opt.patch_size[0],
    #           opt.patch_size[1], opt.patch_size[2], opt.stride_inplane, opt.stride_layer, condition_dict, 1)

