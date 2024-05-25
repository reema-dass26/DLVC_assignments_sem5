#!/usr/bin/env python3

import argparse
import collections
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as v2
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, StepLR
from torchvision.models.segmentation import FCN, fcn_resnet50
from torchvision.models.segmentation.fcn import FCNHead
from tqdm import tqdm

from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.models.segment_model import DeepSegmenter
from dlvc.trainer import ImgSemSegTrainer


def train(args):

    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    train_transform2 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.long, scale=False),
            v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        ]
    )

    val_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_transform2 = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.long, scale=False),
            v2.Resize(size=(64, 64), interpolation=v2.InterpolationMode.NEAREST),
        ]
    )

    train_data = OxfordPetsCustom(
        root=args.files,
        split="trainval",
        target_types="segmentation",
        transform=train_transform,
        target_transform=train_transform2,
        download=True,
    )

    val_data = OxfordPetsCustom(
        root=args.files,
        split="test",
        target_types="segmentation",
        transform=val_transform,
        target_transform=val_transform2,
        download=True,
    )

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
    elif args.model == "deeplabv3_resnet50":
        # TODO (dm) Why does this exist
        pass
        #models.append(DeepSegmenter(deeplabv3_resnet50()))
    else:
        pass
        #models.append(DeepSegmenter(fcn_resnet50()))
        #models.append(DeepSegmenter(deeplabv3_resnet50()))

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    if args.scheduler == "exponential":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == "linear":
        lr_scheduler = LinearLR(optimizer, 30)
    else:
        lr_scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)

    optimizer = AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        amsgrad=args.amsgrad,
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 5

    for model in models:
        print(f"==> Started {model.mname()}")
        model.to(device)
        trainer = ImgSemSegTrainer(
            model,
            optimizer,
            loss_fn,
            lr_scheduler,
            train_metric,
            val_metric,
            train_data,
            val_data,
            device,
            args.epochs,
            model_save_dir,
            batch_size=64,
            val_frequency=val_frequency,
        )
        
        trainer.train()
        # see Reference implementation of ImgSemSegTrainer
        # just comment if not used
        trainer.dispose()


if __name__ == "__main__":
    # fmt: off
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-p", "--pretrained", default="True", choices=["True", "False"]
    )
    args.add_argument(
        "-m", "--model", default="fcn_resnet50", choices=["fcn_resnet50", ]
    )
    args.add_argument(
        "-e", "--epochs", default=100, type=int, help="Number of epochs"
    )
    args.add_argument(
        "-w", "--weight_decay", default=1e-2, type=float, help="Weight decay"
    )
    args.add_argument(
        "-l", "--learning_rate", default=1e-3, type=float, help="Learning rate"
    )
    args.add_argument(
        "-g", "--amsgrad", default=True, action=argparse.BooleanOptionalAction
    )
    args.add_argument(
        "-b", "--batch_size", default=128, type=int, help="Batch size"
    )
    args.add_argument(
        "-s", "--scheduler", default="exponential", choices=["exponential", "linear", "step"]
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    train(args)
    # fmt: on
