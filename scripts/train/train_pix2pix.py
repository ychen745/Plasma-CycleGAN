import sys
from utils.NiftiDataset import *
import utils.NiftiDataset as NiftiDataset
from torch.utils.data import DataLoader
from options.train_options import TrainOptions
# from logger import *
import time
from models import create_model
from utils.visualizer import Visualizer
from utils.earlystop import EarlyStopping

if __name__ == '__main__':

    # -----  Loading the init options -----
    opt = TrainOptions().parse()


    def print_current_losses(epoch, i, losses, t, t_data):
        loss_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_plot.txt')
        log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')

        message = ['(epoch: %d, iters: %d, time: %.3f, data: %.3f)' % (epoch, i, t, t_data)]
        for k, v in losses.items():
            message.append('%s: %.3f' % (k, v))

        print(' '.join(message))
        with open(log_name, "a") as log_file:
            log_file.write('\t'.join(message) + '\n')

        loss_msg = '%s,%.3f,%.3f' %  (epoch, losses['D'], losses['G'])
        with open(loss_name, 'a') as f:
            f.write(loss_msg + '\n')


    # -----  Transformation and Augmentation process for the data  -----
    min_pixel = int(opt.min_pixel * ((opt.patch_size[0] * opt.patch_size[1] * opt.patch_size[2]) / 100))
    trainTransforms = [
                # NiftiDataset.Resample(opt.new_resolution, opt.resample),
                # NiftiDataset.Augmentation(),
                # NiftiDataset.Padding((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2])),
                NiftiDataset.RandomCrop((opt.patch_size[0], opt.patch_size[1], opt.patch_size[2]), opt.drop_ratio, min_pixel),
                ]
    if opt.cross_validation:
        train_set = NiftiDataSet(opt.data_path, True, opt.data_split, opt.fold, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='train')
        # val_set = NiftiDataSet(opt.data_path, True, opt.data_split, opt.fold, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='val')
    else:
        train_set = NiftiDataSet(opt.data_path, opt.data_split, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='train')
        # val_set = NiftiDataSet(opt.data_path, opt.data_split, which_direction='AtoB', transforms=trainTransforms, shuffle_labels=False, mode='val')
    print('lenght train list:', len(train_set))
    # print('lenght val list:', len(val_set))
    train_loader = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=True)  # Here are then fed to the network with a defined batch size
    # val_loader = DataLoader(val_set, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, pin_memory=False)

    # -----------------------------------------------------
    model = create_model(opt)  # creation of the model
    model.setup(opt)
    if opt.epoch_count > 1:
        model.load_networks(opt.epoch_count)
    visualizer = Visualizer(opt)
    total_steps = 0

    if opt.use_earlystop:
        print('using early stop')
        early_stopping = EarlyStopping(patience=opt.patience, verbose=True)

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
                print_current_losses(epoch, epoch_iter, losses, t, t_data)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
