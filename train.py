import math
import os

from tatl_ssl.tatlaug import TATLAugmentations

import torch
from torch import optim
import sys
import time
import json
import torchvision.transforms as T

from pathlib import Path


from tatl_ssl.backbones import ResNetBackbone
from tatl_ssl.tatl import TATL
import tatl_ssl.sslparser as sslparser


from tatl_ssl.dataset import get_dataset_spec
from tatl_ssl.myDataset import myDataset



@sslparser.dataparser

class TrainingArgs:
    "A SSL Trainer"

    train_dir: Path = sslparser.Field(positional=True,help="The training dataset path") 
    checkpoint_dir: Path = Path("checkpoint/")

    backbone_arch: str = sslparser.Field(default="resnet18")
    
    method: str = sslparser.Field(default="tatl")
    optimizer: str = sslparser.Field(default="sgd")

    dataset_id: str = sslparser.Field(default="CSM", choices=["CSM", "BICGSV", "BEAUTY"])
    temperature: float = sslparser.Field(default=0.07, help="Temperature for the nt_xent loss [default=0.07]")

    lambd: float = 0.0002
    n_epochs: int = 100
    batch_size: int = 128
    weight_decay: float = 1e-4
    cosine: bool = sslparser.Field(action="store_true")
    learning_rate_weights: float = 0.2
    learning_rate_biases: float = 0.0048

    num_workers: int = (os.getenv("SLURM_CPUS_PER_TASK", os.cpu_count() // 4))
    is_distributed: bool = False


def adjust_learning_rate(args, optimizer, loader, step):
    if args.cosine:
        max_steps = args.n_epochs * len(loader)
        warmup_steps = 10 * len(loader)
        base_lr = args.batch_size / 512
        if step < warmup_steps:
            lr = base_lr * step / warmup_steps
        else:
            step -= warmup_steps
            max_steps -= warmup_steps
            q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1 - q)
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases
    else:
        lr_decay_steps = [700, 800, 900]
        lr_decay_rate = 0.1

        n_epochs = step // len(loader)
        steps = sum(n_epochs > x for x in lr_decay_steps)

        lr = lr_decay_rate ** steps
        optimizer.param_groups[0]["lr"] = lr * args.learning_rate_weights
        optimizer.param_groups[1]["lr"] = lr * args.learning_rate_biases

def main_worker(device, args: TrainingArgs):

    torch.manual_seed(42)

    if device != "cpu":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True

    dataset_spec = get_dataset_spec(args.dataset_id)
    img_size, crop_size = dataset_spec.size, dataset_spec.crop_size

    augment = TATLAugmentations(
        size=crop_size,
        mean=dataset_spec.mean,
        std=dataset_spec.std,
        )
    img_transform = T.Compose([T.Resize(img_size), T.CenterCrop(crop_size), augment,])
    train_dataset = myDataset(root=args.train_dir, transform=img_transform)


    sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.is_distributed
        else None
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=not args.is_distributed,     
        num_workers=args.num_workers,
        pin_memory=device != "cpu",
        persistent_workers=False,
    )

    backbone = ResNetBackbone(args.backbone_arch)


    model = TATL(
        backbone,
        lambd=0.0002,
        batch_size=args.batch_size,
        h_dim=backbone.out_dim * (4),
    )

    model = model.to(device)

    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{"params": param_weights}, {"params": param_biases}]

    if args.optimizer == "sgd":
        opt = optim.SGD(parameters, 0, momentum=0.9, weight_decay=args.weight_decay)

    model.train()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint():
        if args.is_distributed and args.rank != 0:
            return
        checkpoint_path = args.checkpoint_dir / "checkpoint.pth"
        state_dict = backbone.state_dict()
        torch.save(state_dict, checkpoint_path)
        print(f">> Saved checkpoint at {checkpoint_path}")

    if not args.is_distributed or args.rank == 0:
        stats_file = open(args.checkpoint_dir / "stats.txt", "w", buffering=1)
        start_time = time.time()

        print(
            f">> Training with {len(train_dataset)} images of size {dataset_spec.size}x{dataset_spec.size} "
            f"on {args.num_workers} workers"
        )


    for epoch in range(args.n_epochs):
        batch_loss = 0

        if sampler is not None and (not args.is_distributed or args.rank == 0):
            sampler.set_epoch(epoch)
        for step, ((x1, x2, x3), _) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)

            if not args.is_distributed or args.rank == 0:
                adjust_learning_rate(args, opt, train_loader, step)

            loss = model(x1, x2, x3)       
            batch_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            

            if not args.is_distributed or args.rank == 0:
                if args.is_distributed:
                    model.module.step(step)
                else:
                    model.step(step)
        

        if epoch % 2 == 0 and (not args.is_distributed or args.rank == 0):
            print(f">> [Epoch {epoch}/{args.n_epochs}] loss = {batch_loss:0.4f}")
            sys.stdout.flush()
            print(
                json.dumps(
                    dict(
                        loss=batch_loss,
                        lr=opt.param_groups[0]["lr"],
                        epoch=epoch,
                        time=int(time.time() - start_time),
                    )
                ),
                file=stats_file,
            )


        if epoch % 10 == 0 and (not args.is_distributed or args.rank == 0):
            save_checkpoint()


    if not args.is_distributed or args.rank == 0:
        save_checkpoint()

def main():
    args = sslparser.from_args(TrainingArgs)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    main_worker(device=device, args=args)
    

if __name__ == "__main__":
    main()
