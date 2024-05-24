
import argparse
import os
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

from datetime import datetime
from dlvc.metrics import Accuracy
from dlvc.trainer import ImgClassificationTrainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR

from pathlib import Path

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


    
    if args.model == "fcn_resnet50": #Pretrained model selection for the assignment
        if args.pretrained=='False':
            # models.append(DeepSegmenter(fcn_resnet50(pretrained=False)))
            model= DeepSegmenter(fcn_resnet50(pretrained=False))

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()
            
            train_metric = SegMetrics(classes=train_data.classes_seg)
            val_metric = SegMetrics(classes=val_data.classes_seg)
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
            # models.append(DeepSegmenter(fcn_resnet50(pretrained=True)))
            model=DeepSegmenter(fcn_resnet50(pretrained=True))
            model.eval()  
            val_data_loader = torch.utils.data.DataLoader(val_data,
                                          batch_size=64,
                                          shuffle=False,
                                          num_workers=1)
            num_train_data = len(train_data)
            num_val_data = len(val_data)
            with torch.no_grad():  # Disable gradient calculation during validation
                # Iterate over the validation dataset
                for images, targets in val_data_loader:
                    images, targets = images.to(device), targets.to(device)

                    # Perform inference
                    outputs = model(images)['out']

                    # Evaluate the predictions (you can use appropriate metrics here)
                    # For example, you can compute accuracy, IoU, etc.
                    # You need to implement this part based on your specific requirements

    
            

    # elif args.model =='deeplabv3_resnet50':
  
    #     pass

    # model = DeepSegmenter(...)
    # for model in models:
    #     print(f"==> Started {model.mname()}")
    #     model_save_dir = model_save_dir / model.mname()
    #     model.to(device)
        
    #     optimizer = AdamW(
    #         model.parameters(), 
    #         weight_decay=args.weight_decay,
    #         lr=args.learning_rate, 
    #         amsgrad=args.amsgrad,
    #     )
    #     loss_fn = torch.nn.CrossEntropyLoss()


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
    args.add_argument(
        "-p", "--pretrained", default="False", choices=["True", "False"]
    )
    args.add_argument(
        "-m", "--model", default="resnet", choices=["fcn_resnet50", ]
    )
    
    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    args.gpu_id = 0
    args.num_epochs = 31


    train(args)