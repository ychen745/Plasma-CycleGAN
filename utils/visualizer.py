import numpy as np
import os
import ntpath
import time

class Visualizer():
    def __init__(self, opt):
        self.name = opt.name
        self.opt = opt
        self.saved = False
        self.loss_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_plot.txt')
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

        with open(self.loss_name, 'a') as f:
            f.write('epoch,D_A,G_A,cycle_A,idt_A,D_B,G_B,cycle_B,idt_B\n')

    def reset(self):
        self.saved = False

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, i, losses, t, t_data):
        message = ['(epoch: %d, iters: %d, time: %.3f, data: %.3f)' % (epoch, i, t, t_data)]
        for k, v in losses.items():
            message.append('%s: %.3f' % (k, v))

        print(' '.join(message))
        with open(self.log_name, "a") as log_file:
            log_file.write('\t'.join(message) + '\n')

        loss_msg = '%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' %  (epoch, losses['D_A'], losses['G_A'], losses['cycle_A'], losses['idt_A'], losses['D_B'], losses['G_B'], losses['cycle_B'], losses['idt_B'])
        with open(self.loss_name, 'a') as f:
            f.write(loss_msg + '\n')
