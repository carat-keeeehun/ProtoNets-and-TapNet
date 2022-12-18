import os
import sys
import yaml
import argparse
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import numpy as np
import torch
# import torch.backends.cudnn as cudnn

from dataloader import get_dataloader
from models import ProtoNet, ResNet12
from losses import PrototypicalLoss
from runner import Runner
from utils import margin_of_error

import pickle
import random

import scipy.stats

'''
This code is training and evaluation code of Prototypical Networks for Few-shot Learning.
'''

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to test the model",
    )
    parser.add_argument(
        '--logdir',
        default='runs',
        type=str,
        help='root where to store models, losses and accuracies'
    )
    parser.add_argument('--resume', action="store_true", help="resume train")
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    
    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    new_config.device = device

    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    
    return args, new_config

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def save_checkpoint(state, is_best, args):
    directory = args.logdir
    filename = directory + f"/checkpoint_{state['epoch']}.pth"

    if not os.path.exists(directory):
        os.makedirs(directory)

    if is_best:
        filename = directory + "/model_best.pth"
        torch.save(state, filename)
    else:
        torch.save(state, filename)


def main():
    args, config = parse_args_and_config()

    if config.data.dataset == 'miniImageNet':
        if args.test:
            # test
            test_loader = get_dataloader(config, 'test')
            input_dim = config.data.image_channel
        else:
            train_loader, val_loader = get_dataloader(config, 'train', 'val')
            input_dim = config.data.image_channel
    else:
        if args.test:
            # test
            test_loader = get_dataloader(config, 'test')
            input_dim = config.data.image_channel
        else:
            train_loader, val_loader = get_dataloader(config, 'train', 'val')
            input_dim = config.data.image_channel

    criterion = PrototypicalLoss().to(config.device)
    runner = Runner(args, config)

    if config.model.name == 'protonet':
        model = ProtoNet(input_dim).to(config.device)
        print("ProtoNet loaded")
    elif config.model.name == 'resnet':
        model = ResNet12([64, 128, 256, 512]).to(config.device)
        print("ResNet loaded")
    else:
        pass
    
    if args.test:
        # test
        if "ckpt_dir" in config.model.__dict__.keys():
            ckpt_dir = config.model.ckpt_dir
            checkpoint = torch.load(
                ckpt_dir,
                map_location=config.device,
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc1 = checkpoint['best_acc1']

        loss_list, acc_list = runner.test(test_loader, model, criterion)

        loss, loss_moe = margin_of_error(loss_list)
        acc, acc_moe = margin_of_error(acc_list)
        
        pl_mi = u"\u00B1"

        loss_string = f'{loss:.3f} {pl_mi} {loss_moe:.3f}'
        acc_string = f'{acc:.3f} {pl_mi} {acc_moe:.3f}'

        print(f"loss : {loss_string:^16}")
        print(f"accuracy : {acc_string:^16}")
                
    else:
        # training
        optimizer = torch.optim.Adam(model.parameters(), config.optim.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, gamma=config.optim.lr_scheduler_gamma,
                                                    step_size=config.optim.lr_scheduler_step)

        # cudnn.benchmark = True
        best_acc1 = 0
        if args.resume:
            try:
                checkpoint = torch.load(sorted(glob(f'{args.logdir}/checkpoint_*.pth'), key=len)[-1])
            except Exception:
                checkpoint = torch.load(args.logdir + '/model_best.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']

            print(f"end of loading checkpoint")
        else:
            start_epoch = 1

        print(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        for epoch in range(start_epoch, config.training.epochs + 1):

            train_loss = runner.train(train_loader, model, optimizer, criterion, epoch)

            # is_test = False if epoch % args.test_iter else True
            if epoch % config.training.val_freq or epoch == config.training.epochs or epoch == 1:

                val_loss, acc1 = runner.validate(val_loader, model, criterion, epoch)

                if acc1 >= best_acc1:
                    is_best = True
                    best_acc1 = acc1
                else:
                    is_best = False

                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, is_best, args)

                if is_best:
                    print(f"*** [Best] Acc : {acc1} at {epoch}")

                print(f"[{epoch}/{config.training.epochs}] {train_loss:.3f}, {val_loss:.3f}, {acc1:.3f}, # {best_acc1:.3f}")

            else:
                print(f"[{epoch}/{config.training.epochs}] {train_loss:.3f}")

            scheduler.step()
    

if __name__ == '__main__':
    main()