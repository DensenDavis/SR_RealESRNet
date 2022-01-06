import torch
import os
from os import path as osp
import shutil
from utils import clone_checkpoint
from dataset import Dataset
from models import MWCNN
from PIL import Image
from torchvision.utils import save_image
from torchvision import transforms
from train_ops import TrainLoop
from torch.utils.tensorboard import SummaryWriter
from config import Configuration
cfg = Configuration()

tb_writer = SummaryWriter(cfg.log_dir)


def load_ckpt(model, ckpt_dir):
    if cfg.ckpt_dir is None:
        epoch = 0
        psnr = 0.0
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        return epoch, model, optimizer, psnr
    
    ckpt_path = osp.join(ckpt_dir,cfg.train_mode,'model.pth')

    if not osp.exists(ckpt_path):
        raise Exception(f"Error! Could't find the checkpoint file {ckpt_path}. Change path to None in config if training from scratch.")

    data = torch.load(ckpt_path)
    epoch = data['epoch']
    model.load_state_dict(data['model_state_dict'])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.load_state_dict(data['optimizer_state_dict'])
    psnr = data["psnr"]
    return epoch, model, optimizer, psnr

def save_checkpoint(epoch, model, optimizer, psnr, ckpt_dir):
    if not osp.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "psnr":psnr
    }
    torch.save(save_data, ckpt_dir+os.sep+'model.pth')
    return

def main():
    epoch = 0
    best_epoch = 0
    dataset = Dataset()
    # model_g = RRDBNet(num_in_ch=3,num_out_ch=3,scale=2,num_block=8).to(cfg.device)
    model_g = MWCNN().to(cfg.device)
    ckpt_dir = clone_checkpoint(cfg.ckpt_dir)
    epoch, model_g, optimizer, best_psnr = load_ckpt(model_g, ckpt_dir)
    train_obj = TrainLoop(dataset, model_g, optimizer, epoch)
    print_string = "Train -> L1 loss : {:.5f}, VGG Loss : {:.5f}, Val -> best psnr : {:.3f}, last psnr : {:.3f}, best epoch :{}"

    while(epoch < cfg.n_epochs):
        epoch += 1
        train_obj.train_one_epoch(epoch)
        # save_latest checkpoint
        if not (epoch % cfg.val_freq) == 0: continue
        save_results = (epoch % cfg.display_frequency) == 0
        display_batch = train_obj.run_validation(save_results)

        tb_writer.add_scalar('train_l1_loss', train_obj.mean_train_l1_loss.average(), epoch)  # Average of stepwise loss of that epoch
        tb_writer.add_scalar('train_vgg_loss', train_obj.mean_train_vgg_loss.average(), epoch)
        tb_writer.add_scalar('net_train_loss', (train_obj.mean_train_l1_loss.average()+train_obj.mean_train_vgg_loss.average())/2, epoch)
        tb_writer.add_scalar('val_l1_loss', train_obj.mean_val_l1_loss.average(), epoch)
        tb_writer.add_scalar('val_vgg_loss', train_obj.mean_val_vgg_loss.average(), epoch)
        tb_writer.add_scalar('net_val_loss', (train_obj.mean_val_l1_loss.average()+train_obj.mean_val_vgg_loss.average())/2, epoch)
        tb_writer.add_scalar('mean_val_psnr', train_obj.mean_val_psnr.average(), epoch)

        if(save_results):
            img_dir = osp.join(cfg.log_dir,"vizualization",str(epoch))
            os.makedirs(img_dir, exist_ok=True)
            for i,img in enumerate(display_batch[:cfg.display_samples]):
                save_image(img, img_dir+os.sep+f"{i}.png")

        if best_psnr <= train_obj.mean_val_psnr.average():
            best_epoch = epoch
            best_psnr = train_obj.mean_val_psnr.average()
            save_checkpoint(epoch,model_g,optimizer,best_psnr,ckpt_dir+os.sep+"best")
            # save_best checkpoint

        save_checkpoint(epoch,model_g,optimizer,best_psnr,ckpt_dir+os.sep+"last")

        print(print_string.format(train_obj.mean_train_l1_loss.average(),
        train_obj.mean_train_vgg_loss.average(),
        best_psnr,
        train_obj.mean_val_psnr.average(),
        best_epoch
        ))

main()