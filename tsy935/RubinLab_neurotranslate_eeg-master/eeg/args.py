import argparse

def get_args():
    parser = argparse.ArgumentParser('Train a SeizureNet on TUH data.')
    
    parser.add_argument('--train_batch_size',
                        type=int,
                        default=50,
                        help='Training batch size.')
    parser.add_argument('--num_workers',
                        type=int,
                        default=4,
                        help='Number of sub-processes to use per data loader.')
    parser.add_argument('--save_dir',
                        type=str,
                        default=None,
                        help='Directory to save the outputs and checkpoints.')
    parser.add_argument('--cross_val',
                        default=False,
                        action='store_true',
                        help='To perform cross-validation.')
    parser.add_argument('--num_folds',
                        type=int,
                        default=5,
                        help='Number of folds in cross-validation.')
    parser.add_argument('--model_name',
                        type=str,
                        default='SeizureNet',
                        choices=('SeizureNet'),
                        help='Which model to use.')
    parser.add_argument('--growth_rate',
                        type=int,
                        default=32,
                        help='Growth rate for dense blocks.')
    parser.add_argument('--compression',
                        type=float,
                        default=0.5,
                        help='Compression rate in dense net.')
    parser.add_argument('--drop_rate',
                        type=float,
                        default=0.,
                        help='Dropout rate in dense net.')
    parser.add_argument('--sample_freq',
                        type=float,
                        default=96,
                        help='Sampling frequency for feature extraction.')
    parser.add_argument('--t_window',
                        type=float,
                        default=1,
                        help='Time window for feature extraction.')
    parser.add_argument('--overlap',
                        type=float,
                        default=0.25,
                        help='Overlap proportion for feature extraction.')
    parser.add_argument('--max_checkpoints',
                        type=int,
                        default=3,
                        help='Maximum number of checkpoints to keep on disk.')
    parser.add_argument('--eval_steps',
                        type=int,
                        default=5000 * .05,
                        help='Number of steps between successive evaluations.')
    parser.add_argument('--metric_name',
                        type=str,
                        default='F1',
                        choices=('F1', 'acc', 'loss'),
                        help='Name of dev metric to determine best checkpoint.')
    parser.add_argument('--lr_init',
                        type=float,
                        default='0.01',
                        help='Initial learning rate.')
    parser.add_argument('--l2_wd',
                        type=float,
                        default=5e-4,
                        help='L2 weight decay.')
    parser.add_argument('--num_epochs',
                        type=int,
                        default=200,
                        help='Number of epochs for which to train.')
    parser.add_argument('--max_grad_norm',
                        type=float,
                        default=5.0,
                        help='Maximum gradient norm for gradient clipping.')
    parser.add_argument('--test_batch_size',
                        type=int,
                        default=128,
                        help='Dev/test batch size.')
    parser.add_argument('--metric_avg',
                        type=str,
                        default='weighted',
                        help='weighted, micro or macro.')
    parser.add_argument('--write_outputs',
                        default=False,
                        action='store_true',
                        help='Whether write prediction outputs to csv file.')
                        
 
    args = parser.parse_args()
    
    if args.metric_name == 'loss':
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ('F1', 'acc'):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'.format(args.metric_name))
        
    return args
