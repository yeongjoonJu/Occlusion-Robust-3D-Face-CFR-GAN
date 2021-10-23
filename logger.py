import random
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainStage1Logger(SummaryWriter):
    def __init__(self, logdir):
        super(TrainStage1Logger, self).__init__(logdir)
    
    def log_training(self, coef, rec, reg, align, per, iteration):
        self.add_scalar("Rec", rec, iteration)
        self.add_scalar("Coef", coef, iteration)
        # self.add_scalar("Heat", heat, iteration)
        self.add_scalar("Landmark", align, iteration)
        self.add_scalar("Reg", reg, iteration)
        self.add_scalar("Perceptual", per, iteration)
    
    def log_train_image(self, img_grid, rendered_grid, iteration):
        self.add_image("train_input", img_grid, iteration)
        self.add_image("rendered_input", rendered_grid, iteration)

    def log_validation(self, value, iteration):
        self.add_scalar("val align", value, iteration)


class TrainStage2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(TrainStage2Logger, self).__init__(logdir)
    
    def log_training(self, coef, fm, align, total, iteration):
        self.add_scalar("Coef", coef, iteration)
        self.add_scalar("Align", align, iteration)
        self.add_scalar("Feature_Matching", fm, iteration)
        self.add_scalar("total_loss", total, iteration)
    
    def log_train_image(self, masked_grid, rendered_grid, img_grid, rendered_t_grid, iteration):
        self.add_image("Masked Face", masked_grid, iteration)
        self.add_image("Masked_render", rendered_grid, iteration)
        self.add_image("Original_input", img_grid, iteration)
        self.add_image("Ori_rendered", rendered_t_grid, iteration)

    def log_validation(self, value, iteration):
        self.add_scalar("val coef loss", value, iteration)