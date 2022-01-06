import torch
from torchvision import transforms
import losses
import metrics
from tqdm import tqdm
from degradation import Degradation
from config import Configuration
cfg = Configuration()

class TrainLoop():
    def __init__(self, dataset, model, optimizer,epoch):
        self.device = cfg.device
        self.dataset = dataset
        self.degradation_generator = Degradation()
        self.model = model
        self.optimizer = optimizer
        self.l1_loss = torch.nn.L1Loss()
        self.vgg_loss = losses.VGG19PerceptualLoss()
        self.mean_train_l1_loss = metrics.Mean()
        self.mean_train_vgg_loss = metrics.Mean()
        self.mean_val_l1_loss = metrics.Mean()
        self.mean_val_vgg_loss = metrics.Mean()
        self.mean_val_psnr = metrics.PSNR()


    # def create_loss_objects(self):
    #     loss_objects = []
    #     for loss_type in cfg.losses:
    #         loss_func = getattr(losses, "get_"+loss_type)
    #         loss_objects.append(loss_func())
    #     return loss_objects


    # def calculate_loss(self, y_true, y_pred):
    #     for i in range(len(self.losses)):
    #         loss = cfg.loss_weights[i]*self.losses(y_true,y_pred)
            
    #         net_loss += loss
    #     return 

    def calculate_loss(self, y_pred, y_true):
        l1_loss = cfg.loss_weights["l1_loss"]*self.l1_loss(y_pred, y_true)
        vgg_loss = cfg.loss_weights["vgg_loss"]*self.vgg_loss(y_pred, y_true)
        self.mean_train_l1_loss.update_average(l1_loss)
        self.mean_train_vgg_loss.update_average(vgg_loss)
        return l1_loss+vgg_loss

    def train_step(self, input_batch, gt_batch):
        self.optimizer.zero_grad()
        output_batch = self.model(input_batch)
        loss = self.calculate_loss(output_batch, gt_batch)
        loss.backward()
        self.optimizer.step()
        return

    def train_one_epoch(self, epoch):
        self.mean_train_l1_loss.reset_states()
        self.mean_train_vgg_loss.reset_states()
        pbar = tqdm(self.dataset.train_ds, desc=f'Epoch : {epoch}')
        for data_batch in pbar:
            lr_batch, gt_batch = self.degradation_generator.apply_degradations(data_batch)
            self.train_step(lr_batch, gt_batch)
        return

    def val_step(self, input_batch, gt_batch):
        output_batch = self.model(input_batch)
        batch_psnr = self.mean_val_psnr(output_batch, gt_batch)
        self.mean_val_psnr.update_average(batch_psnr)
        self.mean_val_l1_loss.update_average(cfg.loss_weights["l1_loss"]*self.l1_loss(output_batch, gt_batch))
        self.mean_val_vgg_loss.update_average(cfg.loss_weights["vgg_loss"]*self.vgg_loss(output_batch, gt_batch))
        return output_batch

    def generate_display_samples(self, display_batch, output_batch):
        if display_batch is None:
            display_batch = output_batch
        else:
            display_batch = torch.cat((display_batch, output_batch), axis=0)
        return display_batch

    def run_validation(self, save_results):
        with torch.no_grad():
            self.mean_val_psnr.reset_states()
            display_batch = None
            for i, data_batch in enumerate(self.dataset.val_ds, start=1):
                input_batch = data_batch['input_img'].to(self.device, dtype=torch.float)
                gt_batch = data_batch['gt_img'].to(self.device, dtype=torch.float)
                output_batch = self.val_step(input_batch, gt_batch)
                if(save_results):
                    display_batch = self.generate_display_samples(display_batch, output_batch)
                    if(display_batch.shape[0] >= cfg.display_samples):
                        save_results = False
        return display_batch