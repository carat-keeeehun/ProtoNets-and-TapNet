import os
import sys
import yaml
import argparse
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import numpy as np
import torch
# import torch.backends.cudnn as cudnn

from generators import miniImageNetGenerator, tieredImageNetGenerator
from models2 import TapNet

import scipy.io as sio
import cupy as cp

import chainer.functions as F
from chainer import optimizers
from chainer import cuda
from chainer import serializers

import pickle
import random

import scipy.stats

'''
This code is training and evaluation code of TapNet: Neural Network Augmented with Task-Adaptive Projection for Few-Shot Learning.
'''

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
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
    
    max_iter = 10001
    xp = cp
    
    input_size = config.data.image_channel * config.data.image_size * config.data.image_size

    model = TapNet(
        nb_class_train=config.training.nb_class_train,
        nb_class_test=config.training.nb_class_test,
        input_size=input_size, 
        dimension=config.training.dimension,
        n_shot=config.training.n_shot,
        gpu=3
    )
    
    optimizer = optimizers.Adam(alpha=config.optim.lr_rate, weight_decay_rate=config.optim.wd_rate)
    model.set_optimizer(optimizer)
    
    if config.data.dataset == 'miniImageNet':
        train_generator = miniImageNetGenerator(
                                        data_file='data/miniImageNet/train.npz', 
                                        nb_classes=config.training.nb_class_train,
                                        nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                        max_iter=max_iter, xp=xp)
        
        savefile_name='tapnet/TapNet_miniImageNet_ResNet12_5shot.mat'
        filename_5shot='tapnet/TapNet_miniImageNet_ResNet12_5shot'
        filename_5shot_last='tapnet/TapNet_miniImageNet_ResNet12_last_5shot'
        
    else:
        train_generator = tieredImageNetGenerator(
                                        image_file='data/tieredImageNet/train_images.npz',
                                        label_file='data/tieredImageNet/train_labels.pkl', 
                                        nb_classes=config.training.nb_class_train,
                                        nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                        max_iter=max_iter, xp=xp)
        
        savefile_name='tapnet/TapNet_tieredImageNet_ResNet12.mat'
        filename_5shot='tapnet/TapNet_tieredImageNet_ResNet12'
        filename_5shot_last='tapnet/TapNet_tieredImageNet_ResNet12_last'
        
    
    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0
    
    # training phase
    for t, (images, labels) in train_generator:
        # train
        loss = model.train(images, labels)
        # logging 
        loss_h.extend([loss.tolist()])
        if (t % 50 == 0):
            print("Episode: %d, Train Loss: %f "%(t, loss))
    
        if (t % 500 == 0):                
            print('Evaluation in Validation data')
            if config.data.dataset == 'miniImageNet':
                test_generator = miniImageNetGenerator(
                                                data_file='data/miniImageNet/val.npz', 
                                                nb_classes=config.training.nb_class_train,
                                                nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                                max_iter=600, xp=xp)
            else:
                test_generator = tieredImageNetGenerator(
                                                image_file='data/tieredImageNet/val_images.npz',
                                                label_file='data/tieredImageNet/val_labels.pkl', 
                                                nb_classes=config.training.nb_class_train,
                                                nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                                max_iter=600, xp=xp)
            scores = []                                              
            for i, (images, labels) in test_generator:
                accs = model.evaluate(images, labels)                
                accs_ = [cuda.to_cpu(acc) for acc in accs]
                score = np.asarray(accs_, dtype=int)
                scores.append(score)
            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t=100*np.mean(np.array(scores))
            
            if acc_best < accuracy_t:
                acc_best = accuracy_t
                epoch_best=t
                serializers.save_npz(filename_5shot,model.chain)
                
            accuracy_h_val.extend([accuracy_t.tolist()])
            del(test_generator)
            del(accs)
            del(accs_)
            del(accuracy_t)
            
            print('Evaluation in Test data')
            if config.data.dataset == 'miniImageNet':
                test_generator = miniImageNetGenerator(
                                                data_file='data/miniImageNet/test.npz', 
                                                nb_classes=config.training.nb_class_train,
                                                nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                                max_iter=600, xp=xp)
            else:
                test_generator = tieredImageNetGenerator(
                                                image_file='data/tieredImageNet/val_images.npz',
                                                label_file='data/tieredImageNet/val_labels.pkl', 
                                                nb_classes=config.training.nb_class_train,
                                                nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                                max_iter=600, xp=xp)
            scores = []                                              
            for i, (images, labels) in test_generator:
                accs = model.evaluate(images, labels)                
                accs_ = [cuda.to_cpu(acc) for acc in accs]
                score = np.asarray(accs_, dtype=int)
                scores.append(score)
            print(('Accuracy 5 shot ={:.2f}%').format(100*np.mean(np.array(scores))))
            accuracy_t=100*np.mean(np.array(scores))
            accuracy_h_test.extend([accuracy_t.tolist()])
            del(test_generator)
            del(accs)
            del(accs_)
            del(accuracy_t)
            sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best})
            if len(accuracy_h_val) > 10:
                print('***Average accuracy on past 10 evaluation***')             
                print('Best epoch =',epoch_best,'Best 5 shot acc=',acc_best)
                
            serializers.save_npz(filename_5shot_last,model.chain)
    
        if (t != 0) and (t % config.training.lrstep == 0) and config.training.lrdecay:
            model.decay_learning_rate(0.1)

    
    accuracy_h5=[]

    serializers.load_npz(filename_5shot, model.chain)
    print('Evaluating the best 5shot model...') 
    for i in range(50):
        if config.data.dataset == 'miniImageNet':
            test_generator = miniImageNetGenerator(
                                            data_file='data/miniImageNet/test.npz', 
                                            nb_classes=config.training.nb_class_train,
                                            nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                            max_iter=600, xp=xp)
        else:
            test_generator = tieredImageNetGenerator(
                                            image_file='data/tieredImageNet/val_images.npz',
                                            label_file='data/tieredImageNet/val_labels.pkl', 
                                            nb_classes=config.training.nb_class_train,
                                            nb_samples_per_class=config.training.n_shot + config.training.n_query, 
                                            max_iter=600, xp=xp)

        scores=[]
        for j, (images, labels) in test_generator:
            accs = model.evaluate(images, labels)                
            accs_ = [cuda.to_cpu(acc) for acc in accs]
            score = np.asarray(accs_, dtype=int)
            scores.append(score)
        accuracy_t=100*np.mean(np.array(scores))
        accuracy_h5.extend([accuracy_t.tolist()])
        print(('600 episodes with 15-query accuracy: 5-shot ={:.2f}%').format(accuracy_t))
        del(test_generator)
        del(accs)
        del(accs_)
        del(accuracy_t)   
        sio.savemat(savefile_name, {'accuracy_h_val':accuracy_h_val, 'accuracy_h_test':accuracy_h_test, 'epoch_best':epoch_best,'acc_best':acc_best, 'accuracy_h5':accuracy_h5})
    print(('Accuracy_test 5 shot ={:.2f}%').format(np.mean(accuracy_h5)))

if __name__ == '__main__':
    main()