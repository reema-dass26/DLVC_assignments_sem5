
import argparse
import os
import torch
import torchvision.transforms.v2 as v2
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
from datetime import datetime

# from dlvc.models.class_model import DeepClassifier  # etc. change to your model
# from dlvc.models.cnn import YourCNN
# from dlvc.models.vit import ViT
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
# from dlvc.datasets.cifar10 import CIFAR10Dataset
# from dlvc.datasets.dataset import Subset


# from torchvision.models import resnet18
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR

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

    train_data = OxfordPetsCustom(root="path_to_dataset", 
                            split="trainval",
                            target_types='segmentation', 
                            transform=train_transform,
                            target_transform=train_transform2,
                            download=True)

    val_data = OxfordPetsCustom(root="path_to_dataset", 
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


    models = []
    if args.model == "fcn_resnet50":
        models.append(DeepSegmenter(fcn_resnet50()))
    # elif args.model == "cnn":
    #     models.append(YourCNN())
    # elif args.model == "vit":
    #     models.append(ViT())
    elif args.model =='deeplabv3_resnet50':
        # models.append(DeepSegmenter(deeplabv3_resnet50())) #TODO
        # models.append(YourCNN())
        # models.append(ViT())

    # model = DeepSegmenter(...)
    for model in models:
        print(f"==> Started {model.mname()}")
        model_save_dir = model_save_dir / model.mname()
        model.to(device)
        
        optimizer = AdamW(
            model.parameters(), 
            weight_decay=args.weight_decay,
            lr=args.learning_rate, 
            amsgrad=args.amsgrad,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        train_metric = Accuracy(classes=train_data.classes)
        val_metric = Accuracy(classes=val_data.classes)
        val_frequency = 5

    optimizer = ...
    loss_fn = ...
    
    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    lr_scheduler = ...
    
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

    # see Reference implementation of ImgSemSegTrainer
    # just comment if not used
    trainer.dispose() 

if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Training')
    args.add_argument('-d', '--gpu_id', default='0', type=str,
                      help='index of which GPU to use')
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31


    train(args)