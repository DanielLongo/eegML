import os
import sys
import time
import logging
import argparse
import torch
from torch import nn
import torch.utils.data as data
import torchvision
from torch import optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torchvision.utils import save_image
sys.path.append("../utils/")
sys.path.append("../../add_noise/")
from add_noise_manual import AddNoiseManual
from add_noise_recordings import AddNoiseRecordings
from add_torch_artificats import get_box_artifact

from log import setup_logging_and_results
sys.path.append("../")
from vq_vae.auto_encoder import *
sys.path.append("../../tsy935/RubinLab_neurotranslate_eeg-master/eeg/")
sys.path.append("../../tsy935/RubinLab_neurotranslate_eeg-master/eeg/data/")
from data_loader_estimated import EstimatedDataset
from constants import *
from data.data_utils import *
from data.data_loader import SeizureDataset
from constants import *
import datetime

# for tensorboard
from tensorboardX import SummaryWriter
# Writer will output to ./runs/ directory by default

def get_timestamp():
    now = datetime.datetime.now()
    out = "denoise_" + str(now.month) + "-" + str(now.day) + "_" + str(now.hour) + ":" + str(now.minute)
    return out

save_file = get_timestamp()
writer = SummaryWriter("./runs/" + save_file) 

print("TENSBOARD SAVE DIR", "./runs/" + save_file)

normalize = torchvision.transforms.Normalize(mean= 0, std= 0.1)
# make deterministic
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
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

save_filename = "Reconstruction-a"

def normalize(batch):
    batch = batch - batch.mean()
    batch = batch / batch.std()
    batch = batch / np.abs(batch).max()
    return batch
    
def main(args):
    args = {
        "model": "vqvae",
        "batch_size": 1, # anything larger and runs out of vRAM
        "hidden": 128, # 128,
        "k": 256, #  512,
        "lr": 5e-7, #5e-6,#2e-4,
        "n_recordings" : n_recordings,
        "vq_coef": 2,
        "commit_coef": 2,
        "kl_coef": 1, 
        "dataset": "imagenet",
        "epochs": 250 * 4,
        "cuda": torch.cuda.is_available(),
        "seed": 1,
        "gpus": "1",
        "log_interval": 50,
        "results_dir": "VAE_imagenet",
        "save_name": "first",
        "data_format": "json",
        "num_workers": 4,
        "num_folds": 5,
    }

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

    if args["cuda"]:
        model.cuda()

    print("Number of Parameters in Model:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args["dataset"] == 'imagenet' else 30, 0.5, )

    kwargs = {'num_workers': 1, 'pin_memory': True} if args["cuda"] else {}
    
    train_dataset = SeizureDataset(TRAIN_SEIZURE_FILE, num_folds=args["num_folds"], cross_val=False, split='train', transform=normalize)
    print("len train dataset", len(train_dataset))
    #train_dataset = EstimatedDataset(800, transform=dataset_transforms[args["dataset"]])

    train_loader = data.DataLoader(dataset=train_dataset,
                                    shuffle=True,
                                    batch_size=args["batch_size"],
                                    num_workers=args["num_workers"]
                                    )
    print("train_loader", len(train_loader))

    eval_dataset = SeizureDataset(DEV_SEIZURE_FILE, num_folds=args["num_folds"], cross_val=False, split='dev', transform=normalize) # TODO: enter right split
   # eval_dataset = EstimatedDataset(20, transform=dataset_transforms[args["dataset"]])
    eval_loader = data.DataLoader(dataset=eval_dataset,
                                  shuffle=False,
                                  batch_size=args["batch_size"],
                                  num_workers=args["num_workers"],
                                  )

    print("eval_loader", len(eval_loader))

    print("Save path", save_path)
    for epoch in range(1, args["epochs"] + 1):
        train_losses = train(epoch, model, train_loader, optimizer, args["cuda"], args["log_interval"], save_path, args)
        test_losses = test_net(epoch, model, eval_loader, args["cuda"], save_path, args)
        writer.flush()
        torch.save(model.state_dict(), "saved_models/" + save_filename + ".pt")

    writer.close()
    # torch.save(model.state_dict(), save_path + "/model" )


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args):
    model.train()
    loss_dict = model.latest_losses()
    losses = {k + '_train': 0 for k, v in loss_dict.items()}
    epoch_losses = {k + '_train': 0 for k, v in loss_dict.items()}
    start_time = time.time()
    batch_idx, data = None, None
    sigmoid = torch.nn.Sigmoid()

    noise = get_box_artifact(0, 100, 0, 100, 224)
    if cuda:
        noise = noise.cuda()
    # noise = torch.eye(224, 224)
    
    for batch_idx, (div_spec, _, _) in enumerate(train_loader):
        # if div_spec_clean.shape[0] != args["batch_size"] or div_spec_clean.shape[1] != 9:
            # print(div_spec_clean.shape)
            # continue
            
        # add [:6] because model too big for more than 6 examples per batch
        div_spec = div_spec.view(-1, 3, 224, 224)[:6]
        if cuda:
            div_spec = div_spec.cuda()

        # normalize now sent as transform
        # div_spec_clean = normalize(div_spec)
        # div_spec_noisy = normalize(noise + div_spec)

        div_spec_clean = div_spec
        div_spec_noisy = noise + div_spec


        optimizer.zero_grad()
        outputs = nn.DataParallel(model)(div_spec_noisy)
        loss = model.loss_function(div_spec_clean, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key + '_train'] += float(latest_losses[key])
            epoch_losses[key + '_train'] += float(latest_losses[key])

        cur_iteration = (epoch-1)*len(train_loader) + batch_idx
        
        writer.add_scalar('Train/mse', float(latest_losses['mse'].item()), cur_iteration)
        writer.add_scalar('Train/vq', float(latest_losses['vq'].item()), cur_iteration)
        writer.add_scalar('Train/commitment', float(latest_losses['commitment'].item()), cur_iteration)

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key + '_train'] /= log_interval
            loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
            logging.info('Train Epoch: {epoch} [{batch:5d}/{total_batch} ({percent:2d}%)]   time:'
                         ' {time:3.2f}   {loss}'
                         .format(epoch=epoch, batch=batch_idx * len(div_spec), total_batch=len(train_loader) * len(div_spec),
                                 percent=int(100. * batch_idx / len(train_loader)),
                                 time=time.time() - start_time,
                                 loss=loss_string))
            start_time = time.time()
            for key in latest_losses:
                losses[key + '_train'] = 0
        if batch_idx == (len(train_loader) - 1):
            save_reconstructed_images(div_spec_clean, div_spec_noisy, epoch, outputs[0], save_path, 'reconstruction_train', cur_iteration=cur_iteration)

        if args["dataset"] == 'imagenet' and batch_idx * len(div_spec) > 25000:
            print("BREAKING" * 20)
            break

    for key in epoch_losses:
        if args["dataset"] != 'imagenet':
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
        else:
            epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    logging.info('====> Epoch: {} {}'.format(epoch, loss_string))
    
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args):
    model.eval()
    loss_dict = model.latest_losses()
    losses = {k + '_test': 0 for k, v in loss_dict.items()}
    batch_idx, data = None, None

    # noise = torch.eye(224, 224)
    noise = get_box_artifact(0, 100, 0, 100, 224)
    if cuda:
        noise = noise.cuda()
    
    with torch.no_grad():
         for batch_idx, (div_spec, _, _) in enumerate(test_loader):
            # if data.shape[0] != args["batch_size"] or data.shape[1] != 9:
               #  continue

            div_spec = div_spec.view(-1, 3, 224, 224)[:6]
            if cuda:
                div_spec = div_spec.cuda()

            div_spec_clean = div_spec
            div_spec_noisy = noise + div_spec

            outputs = nn.DataParallel(model)(div_spec_noisy)
            latest_losses = model.latest_losses()
            model.loss_function(div_spec_clean, *outputs)
            latest_losses = model.latest_losses()

            cur_iteration = epoch*len(test_loader) + batch_idx

            writer.add_scalar('Eval/mse', float(latest_losses['mse'].item()), cur_iteration)
            writer.add_scalar('Eval/vq', float(latest_losses['vq'].item()), cur_iteration)
            writer.add_scalar('Eval/commitment', float(latest_losses['commitment'].item()), cur_iteration)

            for key in latest_losses:
                losses[key + '_test'] += float(latest_losses[key])
            if batch_idx == 0:
                save_reconstructed_images(div_spec_clean, div_spec_noisy, epoch, outputs[0], save_path, 'reconstruction_test', cur_iteration=cur_iteration)

            if args["dataset"] == 'imagenet' and batch_idx * len(div_spec) > 1000:
                break

    for key in losses:
        if args["dataset"] != 'imagenet':
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (batch_idx * len(div_spec))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    logging.info('====> Test set losses: {}'.format(loss_string))
    return losses


def save_reconstructed_images(data, noisy, epoch, outputs, save_path, name, cur_iteration=None):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    outputs = outputs.view(batch_size, size[1], size[2], size[3])[:n]
    comparison = torch.cat([data[:n],
                            noisy[:n],
                            outputs])

    save_image(comparison.cpu(),
               os.path.join(save_path, name + '_' + str(epoch) + '.png'), nrow=n, normalize=False)


if __name__ == "__main__":
    main(sys.argv[1:])
