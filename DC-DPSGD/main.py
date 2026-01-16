import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import argparse
import csv
import random
import time
import pandas as pd
import numpy as np

from torch.distributions import Normal, weibull

from opacus.accountants.utils import get_noise_multiplier

from torch.utils.data import Subset, DataLoader, RandomSampler
from load_data import load_tabular_local

from gaussian_svt_domain_partition import GaussianSVTDomainPartition

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from torch.utils.data import TensorDataset, DataLoader

from models.resnet import resnet9, resnet18
#from torchvision.models import vit_l_16, vgg16
from models.cnn import CNN, LeNet
from utils import get_data_loader, get_sigma, restore_param, checkpoint, adjust_learning_rate, process_grad_batch, process_grad
from main_utils import save_pro

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from opacus.accountants import RDPAccountant

parser = argparse.ArgumentParser(description='Differentially Private learning with DP-SGD')

## general arguments
parser.add_argument('--dataset', default='breast_cancer', type=str, help='dataset name')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--sess', default='breast_cancer_cnn', type=str, help='session name')
parser.add_argument('--seed', default=9, type=int, help='random seed') #1-2-3
parser.add_argument('--weight_decay', default=0., type=float, help='weight decay')
parser.add_argument('--batchsize', default=128, type=int, help='batch size')
parser.add_argument('--n_epoch', default=40, type=int, help='total number of epochs')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')
parser.add_argument('--time', default='03092235', type=str, help='time')
parser.add_argument('--save_dir', default='res', type=str, help='save path')



## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--hdp', default=False, type=bool, help='enable hdp-sgd')
parser.add_argument('--clip', default= 0.1, type=float, help='gradient clipping bound') #0.01-1
parser.add_argument('--s_clip', default= 1, type=float, help='gradient clipping bound')

parser.add_argument("--f_epoch", type=bool, default=None)

parser.add_argument('--lr', default= 0.1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--eps', default=8, type=float, help='privacy parameter epsilon')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')

args = parser.parse_args()

print("lr:",args.lr)
print("clip:",args.clip)
print("sclip:",args.s_clip)


assert args.dataset in ['cifar10', 'cifar10-LT', 'svhn', 'mnist', 'fmnist', 'imagenette', 'breast_cancer']

use_cuda = True
best_acc = 0  
accuracy_accountant = []
grad_norm_accountant = []
start_epoch = 0  
batch_size = args.batchsize

if(args.seed != -1): 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

print('==> Preparing data..')

### tabular dataset
if(args.dataset == 'adult'):
    adult_path = "./data/adult"
    X_train, X_test, y_train, y_test = load_tabular_local(args.dataset, adult_path)

    n_training = len(X_train)
    n_test = len(X_test)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)

elif(args.dataset == 'yeast'):
    path = ""
    X_train, X_test, y_train, y_test = load_tabular_local(args.dataset, path)

    n_training = len(X_train)
    n_test = len(X_test)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)

elif(args.dataset == 'product'):
    path = "./data/product/pricerunner_aggregate.csv"
    X_train, X_test, y_train, y_test = load_tabular_local(args.dataset, path)

    n_training = len(X_train)
    n_test = len(X_test)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)
elif(args.dataset == 'bank'):
    bank_path = "./data/bank/bank.csv"
    X_train, X_test, y_train, y_test = load_tabular_local(args.dataset, bank_path)

    n_training = len(X_train)
    n_test = len(X_test)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)
elif(args.dataset == 'credit'):
    credit_path = "./data/credit_card/credit.xls"
    X_train, X_test, y_train, y_test = load_tabular_local(args.dataset, credit_path)

    n_training = len(X_train)
    n_test = len(X_test)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)

elif(args.dataset == 'breast_cancer'):
    X, y = load_breast_cancer(return_X_y=True, as_frame=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train_std = scaler.transform(X_train)
    X_test_std  = scaler.transform(X_test)
    n_training = len(X_train_std)
    n_test = len(X_test_std)
    train_samples, train_labels = None, None
    Xtr = torch.tensor(X_train_std, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test_std, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)
elif(args.dataset == 'android_malware'):

    df = pd.read_csv("./data/malware/TUANDROMD.csv")
    df.columns = [c.strip().lower() for c in df.columns]

    df['label'] = df['label'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['label'])


    if df["label"].dtype == object:
        mapping = {"malware": 1, "goodware": 0, "Malware": 1, "Goodware": 0}
        df["label"] = df["label"].map(mapping)

    y = df['label'].astype(int).values
    X = df.drop(columns=['label'])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_proc = preprocess.fit_transform(X_train_raw)
    X_test_proc  = preprocess.transform(X_test_raw)
    n_training = len(X_train_proc)
    n_test = len(X_test_proc)
    train_samples, train_labels = None, None

    print("Processed shapes:", X_train_proc.shape, X_test_proc.shape)
    Xtr = torch.tensor(X_train_proc, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.long)
    Xte = torch.tensor(X_test_proc, dtype=torch.float32)
    yte = torch.tensor(y_test, dtype=torch.long)

    trainset = TensorDataset(Xtr, ytr)
    testset = TensorDataset(Xte, yte)
    trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True)
    testloader  = DataLoader(testset, batch_size=16, shuffle=False)

### image dataset
if(args.dataset == 'cifar10'):
    trainloader, testloader, n_training, n_test = get_data_loader('cifar10', batchsize = args.batchsize)
    train_samples, train_labels = None, None

elif(args.dataset == 'mnist'):
    trainloader, testloader, trainset, n_training, n_test = get_data_loader('mnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None
elif(args.dataset == 'fmnist'):
    trainloader, testloader, trainset, n_training, n_test = get_data_loader('fmnist', batchsize = args.batchsize)
    train_samples, train_labels = None, None
elif(args.dataset == 'imagenette'):
    trainloader, testloader, n_training, n_test = get_data_loader('imagenette', batchsize = args.batchsize)
    train_samples, train_labels = None, None

print('# of training examples: ', n_training, '# of testing examples: ', n_test)


print('\n==> Computing noise scale for privacy budget (%.1f, %f)-DP'%(args.eps, args.delta))
sampling_prob=args.batchsize/n_training

noise_multiplier = get_noise_multiplier(target_epsilon= args.eps-0.4, target_delta=args.delta, 
            sample_rate= sampling_prob, epochs=args.n_epoch, accountant='rdp')

noise_multiplier_threhold = get_noise_multiplier(target_epsilon= 0.1, target_delta=args.delta, 
            sample_rate= 1, epochs=args.n_epoch, accountant='rdp')
noise_multiplier_query = get_noise_multiplier(target_epsilon= 0.3, target_delta=args.delta, 
            sample_rate= 1, epochs=args.n_epoch, accountant='rdp')
print('noise scale: ', noise_multiplier, 'privacy guarantee: ', args.eps)
print('noise scale_trace: ', noise_multiplier_query)


print('\n==> Creating '+ args.sess +' model instance')
if(args.resume):
    try:
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint_file = './checkpoint/' + args.sess  + '.ckpt'
        checkpoint = torch.load(checkpoint_file)
        #net = resnet20()
        net = CNN(input_dim=1, output_dim=10)
        net.cuda()
        restore_param(net.state_dict(), checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
    except:
        print('resume from checkpoint failed')
else:
    if 'cifar' in args.dataset:
        m1 = torchvision.models.resnet34(weights="IMAGENET1K_V1")
    elif 'breast_cancer' in args.dataset:
        net = nn.Sequential(
            nn.Linear(X.shape[1], 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        net.cuda()
    elif 'android_malware' in args.dataset:
        input_dim = X_train_proc.shape[1]
        net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
        print('using 2-nn-android')
        net.cuda()
    elif args.dataset =='adult' or args.dataset == 'credit' or args.dataset == 'bank':
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
        print('using 2-nn')
        net.cuda()
    elif args.dataset == 'yeast' or args.dataset == 'product':
        net = nn.Sequential(
            nn.Linear(Xtr.shape[1], 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )
        print('using nn-10')
        net.cuda()
    else:
        net = CNN(input_dim=1, output_dim=10)
        net.cuda()

net = extend(net)

num_params = 0
for p in net.parameters():
    num_params += p.numel()

print('total number of parameters: ', num_params/(10**6), 'M')

if(args.private):
    loss_func = nn.CrossEntropyLoss(reduction='sum')
else:
    loss_func = nn.CrossEntropyLoss(reduction='mean')

loss_func = extend(loss_func)

num_params = 0
np_list = []
for p in net.parameters():
    num_params += p.numel()
    np_list.append(p.numel())

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)

#optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    
    train_loss = 0
    correct = 0
    total = 0
    Vk = []
    t0 = time.time()
    steps = n_training//args.batchsize

    if(train_samples == None): # using pytorch data loader for CIFAR10
        loader = iter(trainloader)
    else: # manually sample minibatchs for SVHN
        sample_idxes = np.arange(n_training)
        np.random.shuffle(sample_idxes)

    if(args.private):
        net.eval()
        per_sample_gradients = []  # Collect all per-sample gradients

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = loss_func(outputs, targets)

            net.zero_grad()
            with backpack(BatchGrad()):
                loss.backward()

            # Collect per-sample gradients as flattened tensors
            grads_flat = []
            for p in net.parameters():
                if hasattr(p, "grad_batch") and p.grad_batch is not None:
                    grads_flat.append(p.grad_batch.reshape(inputs.size(0), -1))
            flat_g = torch.cat(grads_flat, dim=1)  # [B, D]

            # Store each sample's gradient as a separate tensor
            for i in range(flat_g.size(0)):
                per_sample_gradients.append(flat_g[i])

        num = len(per_sample_gradients)
        print(f"Total samples: {num}")

        # Initialize Gaussian SVT with domain partition
        svt = GaussianSVTDomainPartition(
            subspace_dim=200,
            tail_proportion=0.1,
            sigma1=noise_multiplier_threhold,
            sigma2=noise_multiplier_query,
            epsilon_tr=0.4,
            delta_tr=args.delta,
            theta=1.0,  # Using theta=1.0 for Weibull distribution
            device='cuda'
        )

        # Run SVT-based private selection
        idx_top, idx_rest = svt.select_heavy_tail_samples(
            gradients=per_sample_gradients,
            threshold=None,  # Auto-compute threshold
            parallel=False  # Set to True for parallel execution if needed
        )

        print(f"Top indices ({len(idx_top)}):")
        print(f"Rest indices ({len(idx_rest)}):")

        top_subset = Subset(trainset, idx_top)
        rest_subset = Subset(trainset, idx_rest)

        top_data_loader = DataLoader(
            top_subset,
            batch_size=args.batchsize,
            sampler=RandomSampler(top_subset, replacement=False),
            num_workers=2, pin_memory=True
            )

        rest_data_loader = DataLoader(
            rest_subset,
            batch_size=args.batchsize,
            sampler=RandomSampler(rest_subset, replacement=False),
            num_workers=2, pin_memory=True
        )   

        top_loader = iter(top_data_loader)
        rest_loader = iter(rest_data_loader)


    net.train()
    
    if epoch % 5 == 0 or epoch==args.n_epoch-1:
        args.f_epoch = True
    for batch_idx in range(steps):
        
        
        if args.private:
            if batch_idx < len(rest_data_loader):
                inputs, targets = next(rest_loader)
                clipping_bs = args.clip * 1
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.1
                
            else:
                inputs, targets = next(top_loader)
                clipping_bs = args.clip * 10
                for param_group in optimizer.param_groups: 
                    param_group['lr'] = 0.1
        else:
            inputs, targets = next(loader)
            for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.1 / args.clip
        

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if(args.private):
            logging = batch_idx % 20 == 0
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            with backpack(BatchGrad()):
                loss.backward()
                ### clip
                process_grad_batch(list(net.parameters()), clipping = clipping_bs)
                
                ### add noise to gradient
                for p in net.parameters():
                    if p.requires_grad == True:
                        shape = p.grad.shape
                        grad_noise = torch.normal(0, noise_multiplier*clipping_bs/args.batchsize, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
                        
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            try:
                for p in net.parameters():
                    del p.grad_batch
            except:
                pass

        optimizer.step()
        step_loss = loss.item()
        if(args.private):
            step_loss /= inputs.shape[0]
        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).float().cpu().sum()
        acc = 100.*float(correct)/float(total)
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')
    return (train_loss/batch_idx, acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    all_correct = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            all_correct += correct_idx.numpy().tolist()
            correct += correct_idx.sum()

        acc = 100.*float(correct)/float(total)
        accuracy_accountant.append(acc)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        ## Save checkpoint.
        if acc > best_acc:
            best_acc = acc
            #checkpoint(net, acc, epoch, args.sess)

    return (test_loss/batch_idx, acc)


print('\n==> Strat training')

for epoch in range(start_epoch, args.n_epoch):
    #lr = adjust_learning_rate(optimizer, args.lr, epoch, all_epoch=args.n_epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    save_pro.save_progress(args, accuracy_accountant, grad_norm_accountant)










