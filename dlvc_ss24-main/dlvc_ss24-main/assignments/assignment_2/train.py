
import argparse
import os
import numpy as np
import torch
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
from pathlib import Path
from dlvc.Fine_tune import ImgSemSegFine_Tune

p = Path('dlvc_ss24-main\\dlvc_ss24-main\\assignments\\assignment_2\\data')

def train(args):

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



    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("==> Using GPU:", torch.cuda.get_device_name())
    else:
        device = torch.device("cpu")
        print("==> CUDA is not available. Using CPU.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    train_metric = SegMetrics(classes=len(train_data.classes_seg))
    val_metric = SegMetrics(classes=len(val_data.classes_seg))

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)
    if args.model == "fcn_resnet50": #Pretrained model selection for the assignment
        if args.pretrained=='False':
            model= DeepSegmenter(fcn_resnet50(pretrained=False))
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            val_frequency = 2
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

            imgSemSegFine_Tune=ImgSemSegFine_Tune()
            imgSemSegFine_Tune.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str, help='index of which GPU to use')
    args.add_argument(
        "-p", "--pretrained", default="True", choices=["True", "False"]
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