"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.mynet as net
#import model.data_loader_stanford_phys as data_loader
#import model.data_loader_weak2 as data_loader
# import model.data_loader_stanford as data_loader

from evaluate import evaluate
from evaluate_model import get_scores, get_scores_test

from torch.optim.lr_scheduler import MultiStepLR
parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', default='experiments_KS_lpch/dump', help="Directory containing params.json")
parser.add_argument('--files_dir', default='file_markers_khaled3/', help="Directory containing txt file of listed names")
parser.add_argument('--dataloader', default=0, help="Dataloader type",type=int)
parser.add_argument('--seed', default=0, help="Seed for cross val",type=int)
parser.add_argument('--scale_ratio', default=1.0, help="Seed for cross val",type=float)

# parser.add_argument('--files_dir', default='file_markers/', help="Directory containing txt file of listed names")

parser.add_argument('--data_dir', default= 'lfs/1/jdunnmon/eeg/EEG/eegdbs/SEC/lpch', help="Directory containing the dataset")
parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")

parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    
    

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
#             if(i>10):
#                 break
            
            # move to GPU if available
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
            # convert to torch Variables
            train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # binarize labels_batch for evaluation
                labels_batch = labels_batch > 0.5
                
                output_batch = output_batch > 0.5
                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data[0]
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data[0])

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    
    return loss_avg()


"""Train the model and evaluate every epoch.

Args:
    model: (torch.nn.Module) the neural network
    train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
    val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
    optimizer: (torch.optim) optimizer for parameters of model
    loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
    metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
    params: (Params) hyperparameters
    model_dir: (string) directory containing config, weights and log
    restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
"""
def train_and_evaluate(model, train_dataloader, val_dataloader, test_dataloader, optimizer, loss_fn, metrics, params, model_dir,scale_ratio,restore_file=None):
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0
    scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.5)

    # for early stopping
    patience = 5
    best_loss = 1e15 #initial start for best loss
    counter = 0
    terminate = False
    
    # choose epoch based on scale, if scale = 1, then epochs = params.epochs
    max_epochs = int(params.num_epochs/scale_ratio)
    
    for epoch in range(max_epochs):
        
        if not(terminate):
            
            scheduler.step()
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, max_epochs))

            # compute number of batches in one epoch (one full pass over the training set)

            train_loss = train(model, optimizer, loss_fn, train_dataloader, metrics, params)

            # Evaluate for one epoch on validation set

            #val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)
            val_metrics,roc_curve_arrays = get_scores(model, loss_fn, val_dataloader, model_dir)


            val_acc = val_metrics['accuracy']
            is_best = val_acc>=best_val_acc

            # Save weights
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict' : optimizer.state_dict()},
                                   is_best=is_best,
                                   checkpoint=model_dir)

            # If best_eval, best_save_path
            if is_best:
                logging.info("- Found new best accuracy")
                best_val_acc = val_acc

                # Save best val metrics in a json file in the model directory
                best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
                utils.save_dict_to_json(val_metrics, best_json_path)
                best_roc_path = os.path.join(model_dir, "best_roc_curve.npy")
                np.save(best_roc_path,roc_curve_arrays)

            # Save latest val metrics in a json file in the model directory
            last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
            utils.save_dict_to_json(val_metrics, last_json_path)
            last_roc_path = os.path.join(model_dir, "last_roc_curve.npy")
            np.save(last_roc_path,roc_curve_arrays)

            # early stopping
            if train_loss > best_loss:
                counter +=1
                if counter >= patience:
                    terminate = True
                    print('Stopped training at epoch %d',epoch)
            else:
                best_loss = train_loss
                counter = 0
            
        
    # TEST MODEL
    check_pnt_path = os.path.join(model_dir,'best.pth.tar') 
    check_pnt = torch.load(check_pnt_path)
    model.load_state_dict(check_pnt['state_dict'])
    
    test_metrics,test_roc_curve_arrays = get_scores_test(model, loss_fn, test_dataloader, model_dir)
    test_json_path = os.path.join(model_dir, "metrics_test_weights.json")
    utils.save_dict_to_json(test_metrics, test_json_path)
    test_roc_path = os.path.join(model_dir, "test_roc_curve.npy")
    np.save(test_roc_path,test_roc_curve_arrays)
    
    


if __name__ == '__main__':
#     data_shape = (2000, 19)
#     data_shape = (3000, 19)
#     os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # Load the parameters from json file
    args = parser.parse_args()
    
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    print("print params: ", params.__dict__)
    
    data_shape = (int(params.clip_len) * 200, 19)
    
    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
#     torch.manual_seed(230)
#     if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    # import the specified dataloader
    if args.dataloader == 0:
        # dataloader for loading 500 annotations
        print('loading dataloader on silver labels')
        import model.data_loader_silver as data_loader
    elif args.dataloader == 1:
        # dataloader for only loading annotations
        print('loading dataloader on annotations')
        import model.data_loader_annot_lpch_cv as data_loader
    elif args.dataloader == 2:
        print('Loading dataloader for gold experiment')
        import model.data_loader_gold2 as data_loader
    elif args.dataloader == 3:
        print('Loading dataloader for on reef + reports + annot')
        import model.data_loader_all as data_loader
    elif args.dataloader == 4:
        print('Loading dataloader on reef for lpch')
        import model.data_loader_reef_scale_cv_lpch as data_loader
    elif args.dataloader == 5:
        print('Loading dataloader on reports for lpch')
        import model.data_loader_reports_scale_cv_lpch as data_loader
    elif args.dataloader == 6:
        print('Loading dataloader for automated pipeline')
        import model.data_loader_automated as data_loader
    else:
        ValueError('Wrong dataloader type')
        
        
    dataloaders = data_loader.fetch_dataloader(['train', 'val','test'], args.data_dir, args.files_dir, params, True, params.clip_len,args.scale_ratio,args.seed)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model and optimizer
#     model = net.Net(params, data_shape).cuda() if params.cuda else net.Net(params, data_shape)
    model = torch.nn.DataParallel(net.Net(params, data_shape)).cuda() if params.cuda else net.Net(params, data_shape)
    print("print model.parameters: ", model.parameters())
    print("weight decay: ", params.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay = params.weight_decay)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, test_dl, optimizer, loss_fn, metrics, params, args.model_dir,args.scale_ratio,
                       args.restore_file)
