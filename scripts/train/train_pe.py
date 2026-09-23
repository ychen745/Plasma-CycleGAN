import sys
from utils.PEDataset import *
import utils.PEDataset as PEDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
from pytorch_msssim import ssim
# import albumentations as A

def normalize_0_1(image):
	# normalize to 0,255
	min_val = np.min(image)
	max_val = np.max(image)

	image = (image - min_val)/(max_val - min_val)

	return image

def normalize_1_1(image):
	# normalize to 0,255
	min_val = np.min(image)
	max_val = np.max(image)

	image = (image - min_val)/(max_val - min_val) * 2 - 1.0

	return image

def normalize_255(image):
	# normalize to 0,255
	min_val = np.min(image)
	max_val = np.max(image)

	image = (image - min_val)/(max_val - min_val) * 255.0

	return image

def structural_similarity_index(image_true, image_generated):
	image_true = torch.tensor(normalize_255(image_true))
	image_generated = torch.tensor(normalize_255(image_generated))
	return ssim(image_generated, image_true, data_range=255)

if __name__ == '__main__':

	# -----  Loading the init options -----
	opt = TrainOptions().parse()

	# -----  Transformation and Augmentation process for the data  -----
	min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))

	trainTransforms = [
				# NiftiDataset.Augmentation(),
				PEDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
				]
	if opt.cross_validation:
		train_set = PEDataSet(opt.data_path, True, opt.data_split, opt.fold, transforms=trainTransforms, mode='train')
		val_set = PEDataSet(opt.data_path, True, opt.data_split, opt.fold, transforms=None, mode='val')
	else:
		train_set = PEDataSet(opt.data_path, False, opt.data_split, transforms=trainTransforms, mode='train')
		val_set = PEDataSet(opt.data_path, False, opt.data_split, transforms=None, mode='val')
	print('lenght train list:', len(train_set))
	train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)
	val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=False, num_workers=opt.workers, pin_memory=True)

	# -----------------------------------------------------
	model = create_model(opt)  # creation of the model
	model.setup(opt)
	if opt.epoch_count > 1:
		model.load_networks(opt.epoch_count)
	visualizer = Visualizer(opt)
	total_steps = 0

	# best_ssim = 0.0
	for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
		epoch_start_time = time.time()
		iter_data_time = time.time()
		epoch_iter = 0

		for i, data in enumerate(train_loader):
			iter_start_time = time.time()
			visualizer.reset()
			total_steps += opt.batch_size
			epoch_iter += opt.batch_size
			model.set_input(data)
			model.optimize_parameters()
			if total_steps % opt.print_freq == 0:
				t = time.time() - iter_start_time
				t_data = iter_start_time - iter_data_time
				losses = model.get_current_losses()
				visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()










