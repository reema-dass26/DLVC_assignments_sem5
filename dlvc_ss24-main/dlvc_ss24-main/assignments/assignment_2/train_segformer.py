#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

import torch
import torchvision.transforms.v2 as v2
import torch.nn as nn
import torch.optim as optim

from torchvision.models.segmentation import FCN, fcn_resnet50

from dlvc.dataset.cityscapes import CityscapesCustom
from dlvc.dataset.oxfordpets import OxfordPetsCustom
from dlvc.metrics import SegMetrics
from dlvc.models.segformer import SegFormer
from dlvc.models.segment_model import DeepSegmenter
from dlvc.trainer import ImgSemSegTrainer


def train_cityscapes(args):

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

    train_data = CityscapesCustom(
        root=args.files,
        split="train",
        mode="fine",
        target_type="semantic",
        transform=train_transform,
        target_transform=train_transform2,
    )
    val_data = CityscapesCustom(
        root=args.files,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=val_transform,
        target_transform=val_transform2,
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DeepSegmenter(fcn_resnet50())
    model.to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        amsgrad=args.amsgrad,
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2  

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    if args.scheduler == "exponential":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == "linear":
        lr_scheduler = LinearLR(optimizer, 30)
    else:
        lr_scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)

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
        args.num_epochs,
        model_save_dir,
        batch_size=args.batch_size,
        val_frequency=val_frequency,
    )
    trainer.train()
    trainer.dispose()


def fine_tune_oxford(args):

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

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = DeepSegmenter(fcn_resnet50(num_classes=3))
    model.to(device)
    model.load(args.model)

    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only update trainable parameters
        weight_decay=args.weight_decay,
        lr=args.learning_rate,
        amsgrad=args.amsgrad,
    )

    loss_fn = nn.CrossEntropyLoss()

    train_metric = SegMetrics(classes=train_data.classes_seg)
    val_metric = SegMetrics(classes=val_data.classes_seg)
    val_frequency = 2  

    model_save_dir = Path("saved_models")
    model_save_dir.mkdir(exist_ok=True)

    if args.scheduler == "exponential":
        lr_scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif args.scheduler == "linear":
        lr_scheduler = LinearLR(optimizer, 30)
    else:
        lr_scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)

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
        args.num_epochs,
        model_save_dir,
        batch_size=args.batch_size,
        val_frequency=val_frequency,
    )
    trainer.train()
    trainer.dispose()



if __name__ == "__main__":
    # fmt: off
    args = argparse.ArgumentParser(description="Training")
    args.add_argument(
        "-f", "--files", default="./data", type=str, help="Location of dataset"
    )
    args.add_argument(
        "-m", "--model", type=str, help="Location of model"
    )
    args.add_argument(
        "-d", "--dataset", choices=["oxford", "city"]
    )
    args.add_argument(
        "-i", "--gpu_id", default="0", type=str, help="index of which GPU to use"
    )
    args.add_argument(
        "-e", "--epochs", default=40, type=int, help="Number of epochs"
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
        "-b", "--batch_size", default=16, type=int, help="Batch size"
    )
    args.add_argument(
        "-s", "--scheduler", default="exponential", choices=["exponential", "linear", "step"]
    )

    if not isinstance(args, tuple):
        args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    if args.dataset == "city":
        train_cityscapes(args)
    else:
        fine_tune_oxford(args)
    # fmt: on