import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import pandas as pd
import utils as utils

from data.data_utils import *
from data.data_loader import SeizureDataset
from constants import *
from args import get_args
from collections import OrderedDict
from json import dumps
from models.SeizureNet import SeizureNet

from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import sys
sys.path.append("/mnt/home2/dlongo/eegML/VQ-VAE-master/")
from vq_vae.auto_encoder import *

# make deterministic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):

    #denoiser = VQ_CVAE(128, k=512, num_channels=3)
    #denoiser.load_state_dict(torch.load("/mnt/home2/dlongo/eegML/VQ-VAE-master/vq_vae/saved_models/train.pt"))     
    #denoiser = torch.no_grad(denoiser)
    #denoiser.cuda()
    #denoiser = nn.DataParallel(denoiser)
    # Get device
    device, args.gpu_ids = utils.get_available_devices()
    args.train_batch_size *= max(1, len(args.gpu_ids))
    args.test_batch_size *= max(1, len(args.gpu_ids))    
    
    # Set random seed
    utils.seed_torch(seed=SEED)
    
    # Get save directories
    train_save_dir = utils.get_save_dir(args.save_dir, training=True)
    args.train_save_dir = train_save_dir
    
    # Save args
    args_file = os.path.join(train_save_dir, ARGS_FILE_NAME)
    with open(args_file, 'w') as f:
         json.dump(vars(args), f, indent=4, sort_keys=True)
    
    # Set up logging and devices   
    log = utils.get_logger(train_save_dir, 'train_denoised')
    tbx = SummaryWriter(train_save_dir)
    log.info('Args: {}'.format(dumps(vars(args), indent=4, sort_keys=True)))
        
    if args.cross_val:
        # Loop over folds
        for fold_idx in range(args.num_folds):        
            log.info('Starting fold {}...'.format(fold_idx))         
    
            # Train
            fold_save_dir = os.path.join(train_save_dir, 'fold_' + str(fold_idx))
            if not os.path.exists(fold_save_dir):
                os.makedirs(fold_save_dir)
            
            # Training on current fold...
            train_fold(args, device, fold_save_dir, log, tbx, cross_val=True, fold_idx=fold_idx)  
            best_path = os.path.join(fold_save_dir, 'best.pth.tar')
        
            # Predict on current fold with best model..
            if args.model_name == 'SeizureNet':
                model = SeizureNet(args)
                
            model = nn.DataParallel(model, args.gpu_ids)
            model, _ = utils.load_model(model, best_path, args.gpu_ids)
         
            model.to(device)
            results = evaluate_fold(model, 
                                   args, 
                                   fold_save_dir, 
                                   device, 
                                   cross_val=True,
                                   fold_idx=fold_idx,
                                   is_test=True, 
                                   write_outputs=True)
                               
            # Log to console
            results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
            print('Fold {} test results: {}'.format(fold_idx, results_str))
            log.info('Finished fold {}...'.format(fold_idx))
    else:
        # no cross-validation
        # Train
        train_fold(args, device, train_save_dir, log, tbx, cross_val=False)
        best_path = os.path.join(train_save_dir, 'best.pth.tar')
        
        if args.model_name == 'SeizureNet':
            model = SeizureNet(args)
                
        model = nn.DataParallel(model, args.gpu_ids)
        model, _ = utils.load_model(model, best_path, args.gpu_ids)
         
        model.to(device)
        results = evaluate_fold(model, 
                               args, 
                               train_save_dir, 
                               device,
                               cross_val=False,
                               fold_idx=None,
                               is_test=True, 
                               write_outputs=True)
        
        # Log to console
        results_str = ', '.join('{}: {:05.2f}'.format(k, v)
                                for k, v in results.items())
        print('Test set prediction results: {}'.format(results_str))
        


def train_fold(args, device, save_dir, log, tbx, cross_val = False, fold_idx = None):
    """
    Perform training and evaluate for the current fold
    """         
    # Define loss function
    class_weights = torch.FloatTensor(CLASS_W)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device) # CrossEntropyLoss includes softmax

    # Get model
    log.info('Building model...')
    if args.model_name == 'SeizureNet':
        model = SeizureNet(args)       
        
    model = nn.DataParallel(model, args.gpu_ids)
    step = 0
    model = model.to(device)
   
    # Get Denoiser
    denoiser = VQ_CVAE(128, k=512, num_channels=3)
    denoiser.load_state_dict(torch.load("/mnt/home2/dlongo/eegML/VQ-VAE-master/vq_vae/saved_models/train.pt"))
    denoiser.eval()
    for param in denoiser.parameters():
        param.requires_grad = False
    denoiser.cuda()
#    denoiser.to(device)
    denoiser = nn.DataParallel(denoiser, args.gpu_ids)
# 
    # To train mode
    model.train()

    # Get saver
    saver = utils.CheckpointSaver(save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # Get optimizer and scheduler
    optimizer = optim.Adam(params=model.parameters(), 
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Get data loader    
    log.info('Building dataset...')
    if cross_val:
        seizure_file = os.path.join('data', 'fold' + str(fold_idx) + '_trainSet_seizure_files.txt')
    else:
        seizure_file = TRAIN_SEIZURE_FILE
        
    train_dataset = SeizureDataset(seizure_file, num_folds=args.num_folds, fold_idx=fold_idx, cross_val=cross_val, split='train')
    train_loader = data.DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=args.train_batch_size,
                                    num_workers=args.num_workers)
    
    # Train
    log.info('Training...')
    steps_till_eval = args.eval_steps
    epoch = step // len(train_dataset)
    while epoch != args.num_epochs:
        epoch += 1
        log.info('Starting epoch {}...'.format(epoch))
        with torch.enable_grad(), \
            tqdm(total=len(train_loader.dataset)) as progress_bar:
            for features, y, _, in train_loader: 
                batch_size = features.shape[0]     
                
                # Setup for forward
                features = features.view(-1, 3, 224, 224) # merge number of dense samples with batch size
                for i in range(0, features.shape[0], 16):
                    features[i:i+16] = denoiser(features[i:i+16])[0]
                features = features.to(device)
                y = y.view(-1) # merge number of dense samples with batch size
                y = y.to(device)
                
                # Zero out optimizer first
                optimizer.zero_grad()
                
                # Forward
                logits = model(features)
                loss = loss_fn(logits, y)
                loss_val = loss.item()
                
                # Backward
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()                

                # Log info
                step += batch_size
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=epoch,
                                         loss=loss_val,
                                         lr=optimizer.param_groups[0]['lr'])
                if cross_val:
                    tbx.add_scalar('fold{}/train/Loss'.format(fold_idx), loss_val, step)
                    tbx.add_scalar('fold{}/train/LR'.format(fold_idx),
                                   optimizer.param_groups[0]['lr'],
                                   step)
                else:
                    tbx.add_scalar('train/Loss', loss_val, step)
                    tbx.add_scalar('train/LR',
                                   optimizer.param_groups[0]['lr'],
                                   step)

                steps_till_eval -= batch_size
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps
                    
                    # Evaluate and save checkpoint
                    log.info('Evaluating at step {}...'.format(step))
                    eval_results = evaluate_fold(model,                                            
                                                args,
                                                save_dir,
                                                device,
                                                cross_val=cross_val,
                                                fold_idx=fold_idx,
                                                is_test=False,
                                                write_outputs=False)
                    best_path = saver.save(step, model, eval_results[args.metric_name], device, eval_results)
                    
                    # Back to train mode
                    model.train()

                    # Log to console
                    results_str = ', '.join('{}: {}'.format(k, v)
                                            for k, v in eval_results.items())
                    log.info('Dev {}'.format(results_str))

                    # Log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in eval_results.items():
                        if cross_val:
                            tbx.add_scalar('fold{}/eval/{}'.format(fold_idx,k), v, step)
                        else:
                            tbx.add_scalar('eval/{}'.format(k), v, step)
        
        # step lr scheduler
        scheduler.step()
        
    return best_path


def evaluate_fold(model, args, save_dir, device, cross_val=False, fold_idx=None, is_test=False, write_outputs=False):
    if cross_val:
        split = 'test'
        seizure_file = os.path.join('data', 'fold' + str(fold_idx) + '_testSet_seizure_files.txt')
    else:
        if is_test:
            split = 'test'
            seizure_file = TEST_SEIZURE_FILE
        else:
            split = 'dev'
            seizure_file = DEV_SEIZURE_FILE
            
    dataset = SeizureDataset(seizure_file, num_folds=args.num_folds, fold_idx=fold_idx, cross_val=cross_val, split=split)
    data_loader = data.DataLoader(dataset=dataset,
                                  shuffle=False,
                                  batch_size=args.test_batch_size,
                                  num_workers=args.num_workers)
    nll_meter = utils.AverageMeter()
    
    # loss function
    class_weights = torch.FloatTensor(CLASS_W)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(device) # CrossEntropyLoss includes softmax
    
    # change to evaluate mode
    model.eval()
    
    y_pred_all = []
    y_true_all = []
    file_name_all = []
    with torch.no_grad(), tqdm(total=len(data_loader.dataset)) as progress_bar:
        for features, y, file_name in data_loader:  
            batch_size = features.shape[0]
                        
            # Setup for forward
            features = features.to(device)
            y = y.view(-1)
            y = y.to(device)
            #print('y shape:{}'.format(y.size()))
            #print('features shape: {}'.format(features.size()))

            # Forward
            logits = model(features) # (batch_size, NUM_CLASSES)
            #print('logits shape: {}'.format(logits.shape))
            #y_pred = F.softmax(logits, dim=1)
            _, y_pred = logits.data.cpu().topk(1, dim=1)
            y_pred = y_pred.view(-1)
            print('y_pred: {}'.format(y_pred))
            print('y_true: {}'.format(y))
            y_pred = y_pred.cpu().numpy()
            y_pred_all.append(y_pred)
            
            loss = loss_fn(logits, y)
            nll_meter.update(loss.item(), batch_size)
            y_true_all.append(y.cpu().numpy())
            file_name_all.extend(file_name)
        
            # Log info
            progress_bar.update(batch_size)

    scores_dict, writeout_dict = utils.eval_dict(y_pred_all, 
                                                 y_true_all, 
                                                 file_name_all, 
                                                 average=args.metric_avg)
    
    results_list = [('Loss', nll_meter.avg),
                    ('acc', scores_dict['acc']),
                    ('F1', scores_dict['F1']),
                    ('recall', scores_dict['recall']),
                    ('precision', scores_dict['precision'])]
    results = OrderedDict(results_list)

    # Write prediction into csv file
    if is_test and write_outputs:        
        df_out = pd.DataFrame(list(writeout_dict.items()), columns=['file','seizure_class'])
        out_file_name = os.path.join(save_dir,split+'_prediction.csv')
        df_out.to_csv(out_file_name, index=False)
        print('Prediction written to {}!'.format(out_file_name))    
            
    return results



if __name__ == '__main__':
    main(get_args())
