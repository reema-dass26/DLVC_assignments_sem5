import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
from dlvc.wandb_logger import WandBLogger
from tqdm import tqdm
from multiprocessing import freeze_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR
import collections
import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from dlvc.metrics import SegMetrics
from dlvc.wandb_logger import WandBLogger
from dlvc.dataset.oxfordpets import OxfordPetsCustom
# Path to dataset
p = Path('dlvc_ss24-main\\dlvc_ss24-main\\assignments\\assignment_2\\data')

class CustomFCNResNet50(nn.Module):
    def __init__(self, backbone, prediction_head):
        super(CustomFCNResNet50, self).__init__()
        self.backbone = backbone
        self.prediction_head = prediction_head

    def forward(self, x):
        x = self.backbone(x)
        x = self.prediction_head(x['out'])
        return x
    def save(self, path: Path, suffix: str = ""):
        model_path = path / f"model_{suffix}.pth"
        torch.save(self.state_dict(), model_path)



# Define transforms
train_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])

train_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
val_transform = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.float32, scale=True),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
val_transform2 = v2.Compose([v2.ToImage(), 
                            v2.ToDtype(torch.long, scale=False),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])
  

train_data = OxfordPetsCustom(root=p, 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

val_data = OxfordPetsCustom(root=p, 
                            split="test",
                            target_types='segmentation', 
                            transform=val_transform,
                            target_transform=val_transform2,
                            download=True)


num_classes = len(val_data.classes_seg)

# Define model
backbone = fcn_resnet50(pretrained=False, pretrained_backbone=True, replace_stride_with_dilation=[False, True, True])
prediction_head = nn.Conv2d(21, num_classes, kernel_size=1)


model = CustomFCNResNet50(backbone, prediction_head)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=1e-4)
lr_scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Dataloaders
train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4)

# Metrics
train_metric = SegMetrics(num_classes)
val_metric = SegMetrics(num_classes)

# Initialize WandB logger
wandb_logger = WandBLogger(enabled=True, model=model, run_name=model.__class__.__name__)

class ImgSemSegFine_Tune():
    def __init__(self):
    
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.lr_scheduler = lr_scheduler

        self.device = device
        
        self.num_epochs = 31
        self.train_metric = train_metric
        self.val_metric = val_metric

        self.subtract_one = isinstance(train_data, OxfordPetsCustom)
        
        self.train_data_loader = torch.utils.data.DataLoader(train_data,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=2)
    
        self.val_data_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=1)
        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)
        self.val_frequency= 5
        model_save_dir = Path("saved_models")
        model_save_dir.mkdir(exist_ok=True)
        self.checkpoint_dir = model_save_dir
        self.wandb_logger = WandBLogger(enabled=True, model=model, run_name='PRETrained_with fine tuning')
        
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
            """
            Training logic for one epoch. 
            Prints current metrics at end of epoch.
            Returns loss, mean IoU for this epoch.

            epoch_idx (int): Current epoch number
            """
            self.model.train()
            epoch_loss = 0.
            self.train_metric.reset()
            
            # train epoch
            for i, batch in tqdm(enumerate(self.train_data_loader), desc="train", total=len(self.train_data_loader)):

                # Zero your gradients for every batch!
                self.optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = batch
                labels = labels.squeeze(1)-1
    
                batch_size = inputs.shape[0] # b x ..?

                # Make predictions for this batch
                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']


                # Compute the loss and its gradients
                loss = self.loss_fn(outputs, labels.to(self.device))
                loss.backward()

                # Adjust learning weights
                self.optimizer.step()

                # Gather metrics
                epoch_loss += (loss.item() * batch_size)
                self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())
        
            self.lr_scheduler.step()
            epoch_loss /= self.num_train_data
            epoch_mIoU = self.train_metric.mIoU()
            
            print(f"______epoch {epoch_idx} \n")
            print(f"Loss: {epoch_loss}")
            print(self.train_metric)

            return epoch_loss, epoch_mIoU

    def _val_epoch(self, epoch_idx:int) -> Tuple[float, float]:
            """
            Validation logic for one epoch. 
            Prints current metrics at end of epoch.
            Returns loss, mean IoU for this epoch on the validation data set.

            epoch_idx (int): Current epoch number
            """
            self.val_metric.reset()
            epoch_loss = 0.
            for batch_idx, batch in tqdm(enumerate(self.val_data_loader), desc="eval", total=len(self.val_data_loader)):
                self.model.eval()
                with torch.no_grad():
                    # get the inputs; data is a tuple of [inputs, labels]
                    inputs, labels = batch
                    labels = labels.squeeze(1)-1
                    batch_size = inputs.shape[0] 

                    # Make predictions for this batch
                    outputs = self.model(inputs.to(self.device))
                    if isinstance(outputs, collections.OrderedDict):
                        outputs = outputs['out']

                    # Compute the loss and its gradients
                    loss = self.loss_fn(outputs, labels.to(self.device))
                    # Gather metrics
                    epoch_loss += (loss.item() * batch_size)
                    self.val_metric.update(outputs.cpu(), labels.cpu())

            epoch_loss /= self.num_val_data
            epoch_mIoU = self.val_metric.mIoU()
            print(f"______epoch {epoch_idx} - validation \n")
            print(f"Loss: {epoch_loss}")
            print(self.val_metric)

            return epoch_loss, epoch_mIoU

    def train(self) -> None:
            """
            Full training logic that loops over num_epochs and
            uses the _train_epoch and _val_epoch methods.
            Save the model if mean IoU on validation data set is higher
            than currently saved best mean IoU or if it is end of training. 
            Depending on the val_frequency parameter, validation is not performed every epoch.
            """
            best_mIoU = 0.
            for epoch_idx in range(self.num_epochs):

                train_loss, train_mIoU = self._train_epoch(epoch_idx)

                wandb_log = {'epoch': epoch_idx}

                # log stuff
                wandb_log.update({"train/loss": train_loss})
                wandb_log.update({"train/mIoU": train_mIoU})

                if epoch_idx % self.val_frequency == 0:
                    val_loss, val_mIoU = self._val_epoch(epoch_idx)
                    wandb_log.update({"val/loss": val_loss})
                    wandb_log.update({"val/mIoU": val_mIoU})

                    if best_mIoU <= val_mIoU:
                        print(f"####best mIou: {val_mIoU}")
                        print(f"####saving model to {self.checkpoint_dir}")
                        # self.model.save(Path(self.checkpoint_dir), suffix="best")
                        best_mIoU = val_mIoU
                    if epoch_idx == self.num_epochs-1:
                        self.model.save(Path(self.checkpoint_dir), suffix="last")

                self.wandb_logger.log(wandb_log)
    
    def dispose(self) -> None:
            self.wandb_logger.finish()  


if __name__ == '__main__':
    freeze_support()
    num_epochs = 20
    imgSemSegTrainer=ImgSemSegTrainer()
    imgSemSegTrainer.train()
    
    # Finish the WandB run
    wandb_logger.finish()