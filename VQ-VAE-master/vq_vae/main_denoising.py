import os
import sys
import time
import logging
import argparse
import torch
from torch import nn
import torch.utils.data
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import save_image
sys.path.append("../utils/")
from log import setup_logging_and_results
# from utils.log import setup_logging_and_results
sys.path.append("../")
from vq_vae.auto_encoder import *
sys.path.append("../../tsy935/RubinLab_neurotranslate_eeg-master/eeg/")
sys.path.append("../../tsy935/RubinLab_neurotranslate_eeg-master/eeg/data/")
from data_loader import SeizureDataset
from constants import *

class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)
    
models = {'imagenet': {'vqvae': VQ_CVAE},
          'cifar10': {'vae': CVAE,
                      'vqvae': VQ_CVAE},
          'mnist': {'vae': VAE,
                    'vqvae': VQ_CVAE}}
datasets_classes = {'imagenet': datasets.ImageFolder,
                    'cifar10': datasets.CIFAR10,
                    'mnist': datasets.MNIST}
dataset_train_args = {'imagenet': {},
                      'cifar10': {'train': True, 'download': True},
                      'mnist': {'train': True, 'download': True}}
dataset_test_args = {'imagenet': {},
                     'cifar10': {'train': False, 'download': True},
                     'mnist': {'train': False, 'download': True},
                     }
dataset_sizes = {'imagenet': (3, 256, 224),
                 'cifar10': (3, 32, 32),
                 'mnist': (1, 28, 28)}

dataset_transforms = {'imagenet': transforms.Compose([#transforms.Resize(128), transforms.CenterCrop(128),
                                                      # transforms.ToTensor(),
                                                      #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]),
                      'cifar10': transforms.Compose([transforms.ToTensor(),
                                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                      'mnist': transforms.ToTensor()}
default_hyperparams = {'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
                       'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
                       'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64}}

def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch

def add_noise(img):
    noise = torch.randn(img.size()) * 0.1
    noise = noise.cuda()
    noisy_img = img + noise
    return noisy_img

def main(args):
    args = {
        "model": "vqvae",
        "batch_size": 16, # 7 * 9 = 63 (close to desired 64)
        "hidden": 128,
        "k": 512 * 2,
        "lr": 2e-4,
        "vq_coef": 1,  # ?
        "commit_coef": 1,  # ?
        "kl_coef": 1,  # ?
        "dataset": "imagenet",
        "epochs": 25,
        "cuda": torch.cuda.is_available(),
        "seed": 1,
        "gpus": "1",
        "log_interval": 50,
        "results_dir": "VAE_imagenet",
        "save_name": "first",
        "data_format": "json"
    }

    #
    # parser = argparse.ArgumentParser(description='Variational AutoEncoders')
    #
    # model_parser = parser.add_argument_group('Model Parameters')
    # model_parser.add_argument('--model', default='vae', choices=['vae', 'vqvae'],
    #                           help='autoencoder variant to use: vae | vqvae')
    # model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
    #                           help='input batch size for training (default: 128)')
    # model_parser.add_argument('--hidden', type=int, metavar='N',
    #                           help='number of hidden channels')
    # model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K',
    #                           help='number of atoms in dictionary')
    # model_parser.add_argument('--lr', type=float, default=None,
    #                           help='learning rate')
    # model_parser.add_argument('--vq_coef', type=float, default=None,
    #                           help='vq coefficient in loss')
    # model_parser.add_argument('--commit_coef', type=float, default=None,
    #                           help='commitment coefficient in loss')
    # model_parser.add_argument('--kl_coef', type=float, default=None,
    #                           help='kl-divergence coefficient in loss')
    #
    # training_parser = parser.add_argument_group('Training Parameters')
    # training_parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10', 'imagenet'],
    #                              help='dataset to use: mnist | cifar10')
    # training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
    #                              help='directory containing the dataset')
    # training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
    #                              help='number of epochs to train (default: 10)')
    # training_parser.add_argument('--no-cuda', action='store_true', default=False,
    #                              help='enables CUDA training')
    # training_parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                              help='random seed (default: 1)')
    # training_parser.add_argument('--gpus', default='0',
    #                              help='gpus used for training - e.g 0,1,3')
    #
    # logging_parser = parser.add_argument_group('Logging Parameters')
    # logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                             help='how many batches to wait before logging training status')
    # logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
    #                             help='results dir')
    # logging_parser.add_argument('--save-name', default='',
    #                             help='saved folder')
    # logging_parser.add_argument('--data-format', default='json',
    #                             help='in which format to save the data')
    # args = parser.parse_args(args)
    # args["cuda"] = not args["no_cuda"] and torch.cuda.is_available()

    lr = args["lr"]  # or default_hyperparams[args["dataset"]]['lr']
    k = args["k"]  # or default_hyperparams[args["dataset"]]['k']
    hidden = args["hidden"] or default_hyperparams[args["dataset"]]['hidden']
    num_channels = dataset_sizes[args["dataset"]][0]

    results, save_path = setup_logging_and_results(args)

    torch.manual_seed(args["seed"])
    if args["cuda"]:
        torch.cuda.manual_seed_all(args["seed"])
        args["gpus"] = [int(i) for i in args["gpus"].split(',')]
        # torch.cuda.set_device(args["gpus"][0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args["seed"])

    model = models[args["dataset"]][args["model"]](hidden, k=k, num_channels=num_channels)
    #model = MyDataParallel(model)
    print("Number of Parameters in Model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    if args["cuda"]:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args["dataset"] == 'imagenet' else 30, 0.5, )

    kwargs = {'num_workers': 1, 'pin_memory': True} if args["cuda"] else {}

    train_loader = torch.utils.data.DataLoader(
        dataset=SeizureDataset(transform=dataset_transforms[args["dataset"]]),
        shuffle=True,
        batch_size=args["batch_size"],
        # transform=dataset_transforms[args["dataset"]],
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=SeizureDataset(file_dir=DEV_SEIZURE_FILE, split="dev", transform=dataset_transforms[args["dataset"]]),  # TODO: add test filename for dataset initialization
        # transform=dataset_transforms[args["dataset"]],
        # dataset=SeizureDataset(transform=dataset_transforms[args["dataset"]]),
        batch_size=args["batch_size"],
        shuffle=True
    )
    print("Save path", save_path)
    for epoch in range(1, args["epochs"] + 1):
        train_losses = train(epoch, model, train_loader, optimizer, args["cuda"], args["log_interval"], save_path, args)
        test_losses = test_net(epoch, model, test_loader, args["cuda"], save_path, args)
        results.add(epoch=epoch, **train_losses, **test_losses)
        for k in train_losses:
             key = k[:-6]
             results.plot(x='epoch', y=[key + '_train', key + '_test'])
        results.save()
        scheduler.step()
    torch.save(model.state_dict(), save_path + "/model" )


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    for batch_idx, (data, _, _) in enumerate(train_loader):
        if data.shape[0] != args["batch_size"] or data.shape[1] != 9:
            continue
        data = data.view(args["batch_size"] * 9, 3, 224, 224)[:4, :, :, :]
        data = normalize(data)
        # print("before shape", data.shape)
        data = torch.nn.functional.pad(input=data, pad=(0,0, 16,16, 0,0, 0,0), mode='constant', value=0)
        if cuda:
            data = data.cuda()
        noisy_data = add_noise(data)
        optimizer.zero_grad()
        outputs = nn.DataParallel(model)(noisy_data)
        loss = model.loss_function(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(data), total_batch=len(train_loader) * len(data),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1) or True:
            save_reconstructed_images(data, noisy_data, epoch, outputs[0], save_path, 'reconstruction_train')
        if args["dataset"] == 'imagenet' and batch_idx * len(data) > 25000:
            break

    for key in epoch_losses:
        if args["dataset"] != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    model.print_atom_hist(outputs[3])
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args):
    print("TESTING")
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    i, data = None, None
    with torch.no_grad():
        print("len of test loader", len(test_loader))
        for i, (data, _, _) in enumerate(test_loader):
            # if data.shape[0] != args["batch_size"] or data.shape[1] != 9:
            # if data.shape[1] != 9:
              #  print("Data", data.shape)
               #  continue
            data = data.view(-1, 3, 224, 224)[:4, :, :, :]
#            print("DATA BEFORE", data)
            data = normalize(data)
#            print("AFTER after", data)
            # print("before shape", data.shape)
            data = torch.nn.functional.pad(input=data, pad=(0,0, 16,16, 0,0, 0,0), mode='constant', value=0)
            if cuda:
                data = data.cuda()
            noisy_data = add_noise(data)
            outputs = nn.DataParallel(model)(noisy_data)
            # outputs = model(data)
            model.loss_function(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if i == 0:
                save_reconstructed_images(data, noisy_data, epoch, outputs[0], save_path, 'reconstruction_test')
            if args["dataset"] == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args["dataset"] != 'imagenet':
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def save_reconstructed_images(data, noisy, epoch, outputs, save_path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            noisy[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=True)


if __name__ == "__main__":
    main(sys.argv[1:])
