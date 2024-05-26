
import argparse
import os
import numpy as np
import torch
from torchvision.models.segmentation import FCN
from torchvision.models.segmentation.fcn import FCNHead
import torchvision.transforms.v2 as v2
import torch.optim as optim
from pathlib import Path
from torchvision.models.segmentation import fcn_resnet50
from dlvc.models.segment_model import DeepSegmenter
from dlvc.dataset.oxfordpets import  OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.trainer import ImgSemSegTrainer
import argparse
import os
import torch
from pathlib import Path
import os
import torch.nn as nn
from tqdm import tqdm
import collections
from datetime import datetime

from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR

from pathlib import Path

p = Path('dlvc_ss24-main\\dlvc_ss24-main\\assignments\\assignment_2\\data')

def train(args):

    train_transform = v2.Compose([v2.ToPILImage(), 
                            v2.ToDtype(torch.float32),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    train_transform2 = v2.Compose([v2.ToPILImage(), 
                            v2.ToDtype(torch.long),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST)])#,
    
    val_transform = v2.Compose([v2.ToPILImage(), 
                            v2.ToDtype(torch.float32),
                            v2.Resize(size=(64,64), interpolation=v2.InterpolationMode.NEAREST),
                            v2.Normalize(mean = [0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])])
    val_transform2 = v2.Compose([v2.ToPILImage(), 
                            v2.ToDtype(torch.long),
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



    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("==> Using GPU:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("==> CUDA is not available. Using CPU.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)

    
    if args.model == "fcn_resnet50": #Pretrained model selection for the assignment
        if args.pretrained=='False':
            model= DeepSegmenter(fcn_resnet50(pretrained=False))
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            
          

            val_frequency = 2

            model_save_dir = Path("saved_models")
            model_save_dir.mkdir(exist_ok=True)

            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1) 
            
            trainer = ImgSemSegTrainer(model, 
                            optimizer,
                            loss_fn,
                            lr_scheduler,
                            train_metric,
                            val_metric,
                            train_data,
                            val_data,
                            device,
                            args.num_epochs, 
                            model_save_dir,
                            batch_size=64,
                            val_frequency = val_frequency)
            trainer.train()


            trainer.dispose() 


        else:
            subtract_one = isinstance(train_data, OxfordPetsCustom)

            backbone=DeepSegmenter(fcn_resnet50(pretrained=True))
            # Extract the encoder part from the backbone (without the final fully connected layer)
            encoder = nn.Sequential(*list(backbone.children())[:-1])
            for param in encoder.parameters():
                param.requires_grad = False
            num_classes = len(val_data.classes)
            model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))

            val_metric.reset()

            val_data_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=1)
            

            def evaluate_model(model, val_data_loader):
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_data_loader:
                        outputs = model(inputs)['out']
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        val_metric.update(outputs.cpu(), labels.cpu())
                        for i in range(num_classes):
                            true_mask = (labels == i)
                            pred_mask = (predicted == i)
                            intersection = (true_mask & pred_mask).sum().item()
                            union = (true_mask | pred_mask).sum().item()
                            if union != 0:
                                class_iou[i] += intersection / union
                            else:
                                class_iou[i] += 1  # If union is 0, set IoU to 1 (best possible score)
                accuracy = correct / total
                class_iou /= len(val_data_loader)
                mean_iou = np.mean(class_iou)
                print(f'Accuracy on test set: {accuracy:.4f}')
                print(f'Mean IoU: {mean_iou:.4f}')
                for i in range(num_classes):
                    print(f'IoU for class {i}: {class_iou[i]:.4f}')
                           
            # Evaluate the model
            evaluate_model(model, val_data_loader)

    
    # optimizer = ...
    # loss_fn = ...
    
    # train_metric = SegMetrics(classes=train_data.classes_seg)
    # val_metric = SegMetrics(classes=val_data.classes_seg)
    # val_frequency = 2

    # model_save_dir = Path("saved_models")
    # model_save_dir.mkdir(exist_ok=True)

    # lr_scheduler = ...
    
    # trainer = ImgSemSegTrainer(model, 
    #                 optimizer,
    #                 loss_fn,
    #                 lr_scheduler,
    #                 train_metric,
    #                 val_metric,
    #                 train_data,
    #                 val_data,
    #                 device,
    #                 args.num_epochs, 
    #                 model_save_dir,
    #                 batch_size=64,
    #                 val_frequency = val_frequency)
    # trainer.train()

    # # see Reference implementation of ImgSemSegTrainer
    # # just comment if not used
    # trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    args.add_argument(
        "-p", "--pretrained", default="False", choices=["True", "False"]
    )
    args.add_argument(
        "-m", "--model", default="fcn_resnet50", choices=["fcn_resnet50", ]
    )
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31


    train(args)