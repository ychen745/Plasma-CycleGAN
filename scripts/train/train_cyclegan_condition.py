import sys
from utils.ConditionDataset import *
import utils.ConditionDataset as ConditionDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
from pytorch_msssim import ssim

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
				# ConditionDataset.Resample(opt.new_resolution, opt.resample),
				# ConditionDataset.Augmentation(),
				# ConditionDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
				ConditionDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
				]
	
	if opt.cross_validation:
		train_set = ConditionDataSet(opt.data_path, True, opt.data_split, opt.fold, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='train', condition=opt.condition)
		val_set = ConditionDataSet(opt.data_path, True, opt.data_split, opt.fold, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='val', condition=opt.condition)
	else:
		train_set = ConditionDataSet(opt.data_path, False, opt.data_split, 0, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='train', condition=opt.condition)
		val_set = ConditionDataSet(opt.data_path, False, opt.data_split, 0, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='val', condition=opt.condition)
	print('lenght train list:', len(train_set))
	train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
	val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

	# -----------------------------------------------------
	model = create_model(opt)  # creation of the model
	model.setup(opt)
	if opt.epoch_count > 1:
		model.load_networks(opt.epoch_count)
	visualizer = Visualizer(opt)
	total_steps = 0

	best_ssim = 0.0
	
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

		batch_ssim = []
		with torch.no_grad():
			model.eval()
			for i, data in enumerate(val_loader):
				# forward pass: compute predicted outputs by passing inputs to the model
				model.set_input(data)
				# model.forward()
				real_pet = model.real_B
				fake_pet = model.netG_A(model.real_A, model.condition, True)
				batch_ssim.append(structural_similarity_index(real_pet.detach().cpu().numpy(), fake_pet.detach().cpu().numpy()))
				
		if np.mean(batch_ssim) > best_ssim:
			best_ssim = np.mean(batch_ssim)
			print('Current best epoch:', epoch)
			model.save_networks(epoch, best=True)

		model.train()

		if epoch % opt.save_epoch_freq == 0:
			print('saving the model at the end of epoch %d, iters %d' %
				  (epoch, total_steps))
			model.save_networks('latest')
			model.save_networks(epoch)

		print('End of epoch %d / %d \t Time Taken: %d sec' %
			  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
		model.update_learning_rate()










