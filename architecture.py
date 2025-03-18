import os
import numpy as np
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
from torch.optim import lr_scheduler
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics import F1Score as F1

import timm
import matplotlib.pyplot as plt

# define the model class
class EfficientNet(pl.LightningModule):
    
    def __init__(self, output_len=5):
        super(EfficientNet, self).__init__()
        
        # create an instance of the EfficientNet model
        self.model = timm.create_model('efficientnetv2_rw_s', num_classes = output_len, pretrained=True)
        
        # Define metrics for each stage
        self.train_acc = Accuracy(task='multiclass', num_classes=5)
        self.train_precision = Precision(task='multiclass', num_classes=5)
        self.train_recall = Recall(task='multiclass', num_classes=5)
        self.train_f1 = F1(task='multiclass', num_classes=5)
        
        self.val_acc = Accuracy(task='multiclass', num_classes=5)
        self.val_precision = Precision(task='multiclass', num_classes=5)
        self.val_recall = Recall(task='multiclass', num_classes=5)
        self.val_f1 = F1(task='multiclass', num_classes=5)
        
        self.test_acc = Accuracy(task='multiclass', num_classes=5)
        self.test_precision = Precision(task='multiclass', num_classes=5)
        self.test_recall = Recall(task='multiclass', num_classes=5)
        self.test_f1 = F1(task='multiclass', num_classes=5)
    # forward pass function
    def forward(self,x):
        out = self.model(x) # pass the input through the model
        return out
    # training step function
    def training_step(self, batch, batch_idx):
        input = batch[0][0]
        target = batch[1]
        pred = self.model(input)
        loss = F.cross_entropy(pred.float(), target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Update metrics
        self.train_acc(pred, target)
        self.train_precision(pred, target)
        self.train_recall(pred, target)
        self.train_f1(pred, target)

        
        return loss

    # validation step function
    def validation_step(self, batch, batch_idx):
        input = batch[0][0]
        target = batch[1]
        pred = self.model(input)
        # compute the loss using cross entropy loss function
        loss = F.cross_entropy(pred.float(), target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        self.val_acc(pred, target)
        self.val_precision(pred, target)
        self.val_recall(pred, target)
        self.val_f1(pred, target)

        return loss
        
    # testing step function
    def test_step(self, batch, batch_idx):
        input = batch[0][0]
        target = batch[1]
        pred = self.model(input)
        # compute the loss using cross entropy loss function
        loss = F.cross_entropy(pred.float(), target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Update metrics
        self.test_acc(pred, target)
        self.test_precision(pred, target)
        self.test_recall(pred, target)
        self.test_f1(pred, target)
        return loss

    def on_train_epoch_end(self):
        # log epoch metric
        self.log('train_acc_epoch', self.train_acc.compute(), sync_dist=True)
        self.log('train_precision_epoch', self.train_precision.compute(), sync_dist=True)
        self.log('train_recall_epoch', self.train_recall.compute(), sync_dist=True)
        self.log('train_f1_epoch', self.train_f1.compute(), sync_dist=True)

        # reset metrics
        self.train_acc.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self):
        # log epoch metric
        self.log('val_acc_epoch', self.val_acc.compute(), sync_dist=True)
        self.log('val_precision_epoch', self.val_precision.compute(), sync_dist=True)
        self.log('val_recall_epoch', self.val_recall.compute(), sync_dist=True)
        self.log('val_f1_epoch', self.val_f1.compute(), sync_dist=True)

        # reset metrics
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        # log epoch metric
        self.log('test_acc_epoch', self.test_acc.compute(), sync_dist=True)
        self.log('test_precision_epoch', self.test_precision.compute(), sync_dist=True)
        self.log('test_recall_epoch', self.test_recall.compute(), sync_dist=True)
        self.log('test_f1_epoch', self.test_f1.compute(), sync_dist=True)

        # reset metrics
        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()


    # define the optimizer and learning rate scheduler
    def configure_optimizers(self):
        # create an instance of the AdamW optimizer
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-2) # default: lr=1e-3
        # sch = lr_scheduler.StepLR(opt, step_size=10, gamma=0.3) # every epoch by default
        # return ({'optimizer': opt, 'lr_scheduler':sch})
        
        # create a learning rate scheduler that decreases the learning rate every 10 epochs by a factor of 0.3
        sch = {'scheduler': lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)}
        return [opt], [sch]

class SpectralSpatialNet(EfficientNet):
    # initialization function
    def __init__(self, num_classes=5, output_len: int=None):
        if output_len is None:
            super(SpectralSpatialNet, self).__init__(num_classes)
            output_len = num_classes
        else:
            super(SpectralSpatialNet, self).__init__(output_len)

        self.spectral_feature_extractor = nn.Sequential(
            nn.Linear(303, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_len),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2*output_len, 2*output_len),
            nn.ReLU(),
            nn.Linear(2*output_len, 5),
        )
        self.img_ln = nn.LayerNorm(output_len)
        self.spectral_ln = nn.LayerNorm(output_len)

        self.img_bn = nn.BatchNorm1d(output_len)
        self.spectral_bn = nn.BatchNorm1d(output_len)

    # forward pass function
    def forward(self,x):
        img, mask, spectra = x

        # efficient_net extraction
        img_features = self.model(img)
        # return img_features
        # spectral extraction
        spectral_features = self.spectral_feature_extractor(spectra)

        # normalize features from each branch
        img_features = self.img_bn(img_features)
        spectral_features = self.spectral_bn(spectral_features)

        # concat
        combined_features = torch.cat((img_features, spectral_features), dim=1)
        out = self.classifier(combined_features)
        return out
    
    def training_step(self, batch, batch_idx):
        data, target = batch  # data is (img, mask, spectra), target is class label
        pred = self(data)  # Use forward method which processes the tuple
        loss = F.cross_entropy(pred, target)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Update metrics
        self.train_acc(pred, target)
        self.train_precision(pred, target)
        self.train_recall(pred, target)
        self.train_f1(pred, target)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        loss = F.cross_entropy(pred, target)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Update metrics
        self.val_acc(pred, target)
        self.val_precision(pred, target)
        self.val_recall(pred, target)
        self.val_f1(pred, target)
        return loss
        
    def test_step(self, batch, batch_idx):
        data, target = batch
        pred = self(data)
        loss = F.cross_entropy(pred, target)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Update metrics
        self.test_acc(pred, target)
        self.test_precision(pred, target)
        self.test_recall(pred, target)
        self.test_f1(pred, target)
        return loss
    
    def configure_optimizers(self):
        # create an instance of the AdamW optimizer
        opt = torch.optim.AdamW([
            {'params': self.model.parameters()},
            {'params': self.spectral_feature_extractor.parameters(), 'lr': 1e-4},
            {'params': self.classifier.parameters(), 'lr': 1e-4}
        ], lr=1e-3, weight_decay=1e-2)
    
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # decreases the learning rate every 10 epochs by a factor of 0.3
        sch = {'scheduler': lr_scheduler.StepLR(opt, step_size=10, gamma=0.3)}
        return [opt], [sch]