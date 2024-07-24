import os
import torch
import argparse
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('-data', type=str, default='fmnist') #Can be synthetic, emnist, fmnist, cifar10 or shakespeare
parser.add_argument('-availability', type=str, default='uniform')  #always, uniform
parser.add_argument('-pretrained-model', type=str, default='')
parser.add_argument('-out', type=str, default=None)

parser.add_argument('-lr', type=float, default=0.1)
# parser.add_argument('-lr-global', type=float, default=1.0)
parser.add_argument('-minibatch', type=int, default=32)

parser.add_argument('-include-gradient-threshold', type=float, default=0.4)
parser.add_argument('-include-gradient-threshold_later', type=float, default=0.1)

parser.add_argument('-lr-warmup', type=float, default=0.1) #The warmup rounds include only priority clients
parser.add_argument('-iters-warmup', type=int, default=500)

parser.add_argument('-eps-drop', type=int, default = 500) #After how long does epsilon become smaller (So you only use priority clients)

parser.add_argument('-iters-total', type=int, default=1000)
parser.add_argument('-seeds', type=str, default='1,2,3,4,5')  # e.g., 1,2,3

parser.add_argument('-iters-per-round', type=int, default=5) #E-value
parser.add_argument('-iters-per-eval', type=int, default=5) #How often do we evaluate, default E = 5

# parser.add_argument('-similarity', type=float, default=0.05)
# parser.add_argument('-similarity_w', type=float, default=0.0)
# parser.add_argument('-disconnect', type=int, default=100)
parser.add_argument('-total-clients', type=int, default=100)
# parser.add_argument('-total-workers_w', type=int, default=10)
parser.add_argument('-sampled-fraction-pr', type=float, default=0.5) #Fraction of priority clients included per round, only used with uniform
parser.add_argument('-sampled-fraction-fr', type=float, default=0.5) #Fraction of free clients included per round, only used with uniform
# parser.add_argument('-temp-worker-size', type=int, default = 2500) #Size of partioned data for workers

parser.add_argument('-priorityfrac', type=float, default = 0.08) #Fraction of clients that are priority clients, only used for non-shakespeare data

parser.add_argument('-gpu', type=int, default=1)  # 1 - use GPU if available; 0 - do not use GPU
parser.add_argument('-cuda-device', type=int, default=0)

# parser.add_argument('-permute', type=int, default=1)
# parser.add_argument('-train_split', type=float, default = 0.5)

parser.add_argument('-save-checkpoint', type=int, default=1)
parser.add_argument('-iters-checkpoint', type=int, default=1500)

# parser.add_argument('-wait-all', type=int, default=0)   # specifies whether to wait for all, after warm up
parser.add_argument('-full-batch', type=int, default=0)  # specifies whether to use full batch, after warm up

# parser.add_argument('-p-value', type=int, default=-1)  # Use (active_rounds + inactive_rounds) if < 0
parser.add_argument('-method', type=str, default='None,FedAVG,FedALIGN,Local') #What methods to use seperated by a comma

parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--iid', type=int, default=0)
parser.add_argument('--label_noise_factor', type=float, default=2.5)
parser.add_argument('--random_data_fraction_factor', type=float, default=1.0)
parser.add_argument('--label_noise_skew_factor', type=float, default=1.0)
parser.add_argument('--random_data_fraction_skew_factor', type=float, default=1.0)
parser.add_argument('-f')
args = parser.parse_args()

print(', '.join(f'{k}={v}' for k, v in vars(args).items()))

config_dict = {
    'alpha': args.alpha,
    'beta': args.beta,
    'iid': args.iid,
    'label_noise_factor': args.label_noise_factor,
    'random_data_fraction_factor': args.random_data_fraction_factor,
    'label_noise_skew_factor': args.label_noise_skew_factor,
    'random_data_fraction_skew_factor': args.random_data_fraction_skew_factor
}


method = args.method
use_gpu = bool(args.gpu)
use_gpu = use_gpu and torch.cuda.is_available()
device = torch.device('cuda:' + str(args.cuda_device)) if use_gpu else torch.device('cpu')

# use_permute = bool(args.permute)
# wait_for_all = bool(args.wait_all)
use_full_batch = bool(args.full_batch)
iter_stop = args.eps_drop
inclusion_threshold = args.include_gradient_threshold
inclusion_threshold_later = args.include_gradient_threshold_later
# use_global_update = bool(args.lr_global > 1.0)
# assert (args.lr_global >= 1.0)

save_checkpoint = bool(args.save_checkpoint)
iters_checkpoint = args.iters_checkpoint

if args.data == 'fmnist':
    dataset = 'FashionMNIST'
    model_name = 'ModelFMnist'
elif args.data == 'emnist':
    dataset = 'EMNIST'
    model_name = 'ModelEMnist'
elif args.data == 'cifar' or args.data == 'cifar10':
    dataset = 'CIFAR10'
    model_name = 'ModelCifar10'
elif args.data == 'synthetic':
    dataset = 'synthetic'
    model_name = 'Synthetic'
elif args.data == 'shakespeare':
    dataset = 'shakespeare'
    model_name = 'ModelShakespeare'
else:
    raise Exception('Unknown data name')

max_iter = args.iters_total

simulations_str = args.seeds.split(',')
simulations = [int(i) for i in simulations_str]

methods = args.method.split(',')

dataset_file_path = os.path.join(os.path.dirname(__file__), 'data_files')

if args.availability == 'uniform':
    uniform = True
else:
    uniform = False


subset_size_priority = args.priorityfrac
subset_size_free = 1 - args.priorityfrac

n_nodes = args.total_clients
pr_nodes = int(n_nodes*subset_size_priority) #split into priority nodes and free nodes
fr_nodes = n_nodes - pr_nodes #same
pr_round = int(args.sampled_fraction_pr*pr_nodes) #Only used if uniform is true
fr_round = int(args.sampled_fraction_fr*fr_nodes) #Only used if uniform is true
step_size_local = args.lr  # learning rate of clients
# step_size_global = args.lr_global

# n_workers = args.total_workers_w

step_size_warmup = args.lr_warmup
iters_warmup = args.iters_warmup

batch_size_train = args.minibatch
batch_size_eval = 256

iters_per_round = args.iters_per_round  # number of iterations in local training
min_iters_per_eval = args.iters_per_eval

results_file = args.out
save_model_file =  'yo.model'
if args.pretrained_model != '':
    load_model_file = args.pretrained_model
else:
    load_model_file = None

if dataset == 'CIFAR10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])
else:
    transform_train = None