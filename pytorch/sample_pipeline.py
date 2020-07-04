import pytorch_lightning as pl
import torch
import torchvision
print("Pytorch Lightning Version : ",pl.__version__)
print("Pytorch  Version : ",torch.__version__)
print("torchvision Version : ",torchvision.__version__)

from torch import nn
import  torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor,Normalize
import os
import pandas as pd
from imgaug import augmenters as iaa
import numpy as np
from custom_loop.data import CSVDataset
import json

###
from torchvision.models import resnet34, resnet18 , wide_resnet50_2
from torch.optim.lr_scheduler import OneCycleLR,CosineAnnealingLR,CosineAnnealingWarmRestarts,CyclicLR,ReduceLROnPlateau
from torch.optim import Adam, AdamW, SGD

# CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
# ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=True, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
# OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', cycle_momentum=True, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0, last_epoch=-1)
# CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)
from pytorch_lightning.loggers.wandb import WandbLogger



seq_cutout = iaa.SomeOf(3, [
    iaa.Sequential([
        # Crop and Pad Mechanism
        iaa.Pad(percent=0.125, keep_size=False),
        iaa.Crop(percent=0.10, keep_size=False)
    ]),

    iaa.ContrastNormalization((0.8, 1.2)),
    iaa.Grayscale(alpha=(0.93, 1.0)),
    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),

    # Flip Right and Left
    iaa.Fliplr(0.5),
    # Rotate 5 Degrees left and right
    iaa.Affine(rotate=10),
    # Similar to Cutout
    iaa.Dropout(p=(0.25)),
], name='Cutout')  # apply augmenters in random order


fn ='config/resnet18_num_0.json'

config= dict(json.load(open(fn,'r')))

aug_func = seq_cutout.augment
BASE_DIR='tiny-imagenet-200'
val_df=pd.read_csv('tiny-imagenet-200/val_annotations.txt',sep='\t',header=None,usecols=[0,1], names=['filename', 'class'])

import pytorch_lightning as pl


model_dict ={
    'resnet18':512,
    'resnet34':512,
    'resnet50':2048,
    'resnet101':2048,
    'resnet152':2048
}




class TinyImagenetModel(pl.LightningModule):
    def __init__(self,root_dir,df,config,augmentation_func=None):
        super(TinyImagenetModel, self).__init__()
        self.root_dir = root_dir
        self.batch_size = config['batch_size']
        self.use_aug =config['use_aug']
        self.optimizer = config['optimizer']
        self.weight_decay = config['weight_decay']
        self.initial_lr = config['lr']
        self.model= self.prep_model(config['model'])
        self.df= df
        # Configure optimizer
        if self.use_aug and augmentation_func:
            self.transforms = Compose([augmentation_func,ToTensor()])
        else:
            self.transforms = Compose([ToTensor()])


    def prep_model(self, model_arch):
        print(f"Setting up Model : {model_arch}")
        model = eval(model_arch)()
        model.fc = nn.Linear(model_dict[model_arch], 200)
        return model

    def forward(self,x):
        return self.model(x)


    def train_dataloader(self):
        self.train_ds = ImageFolder(os.path.join(self.root_dir, 'train'), transform=self.transforms)
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True,drop_last=True, num_workers=0)

    def val_dataloader(self):
        self.val_ds = CSVDataset(os.path.join(self.root_dir, 'val'), self.df,label_col='class', transform=self.transforms)
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, pin_memory=True,drop_last=True, num_workers=0)

    def configure_optimizers(self):
        return eval(self.optimizer)(self.parameters(),lr=self.initial_lr,weight_decay=self.weight_decay)

        # optimizer = eval(self.optimizer)(self.parameters(),lr=self.initial_lr,weight_decay=self.weight_decay)
        # scheduler = eval(self.scheduler)(optimizer, )
        # return [optimizer], [scheduler]


    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

experiments_names = '_'.join([ str(v) for v in config.values()])
print(experiments_names)
wandb_logger = WandbLogger(name=experiments_names,project='tiny-imagenet')

model = TinyImagenetModel(BASE_DIR,df=val_df,config=config,augmentation_func=seq_cutout.augment)

trainer = pl.Trainer(max_epochs=50,gpus=1,logger=wandb_logger)


trainer.fit(model)



