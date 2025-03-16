import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import ftrain


parser = argparse.ArgumentParser(description='Cross-Modal attention fusion')
parser.add_argument('-f', default='', type=str)

parser.add_argument('--model', type=str, default='cross-modal',
                    help='name of the model to use (Transformer, etc.)')
parser.add_argument('--v1only', action='store_true', default=False,
                    help='use the crossmodal fusion into v1 (default: False)')
parser.add_argument('--v2only', action='store_true', default=False,
                    help='use the crossmodal fusion into v2 (default: False)')
parser.add_argument('--ponly', action='store_true', default=False,
                    help='use the crossmodal fusion into p (default: False)')

parser.add_argument('--dataset', type=str, default='deap9class0',
                    help='dataset to use (default: deap9class0)')
parser.add_argument('--data_path', type=str, default='data',
                    help='path for storing the dataset')
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_v1', type=float, default=0.1,
                    help='attention dropout (for vision1)')
parser.add_argument('--attn_dropout_v2', type=float, default=0.1,
                    help='attention dropout (for vision2)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.1,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout')
parser.add_argument('--nlevels', type=int, default=5,
                    help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--batch_size', type=int, default=40, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=300,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=20,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')
parser.add_argument('--log_interval', type=int, default=30,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')
parser.add_argument('--name', type=str, default='fusion',
                    help='name of the trial')
parser.add_argument('--check', action='store_true', default=False,
                    help='name of the trial')
args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.ponly + args.v1only + args.v2only

if valid_partial_mode == 0:
    args.ponly = args.v1only = args.v2only = True
if valid_partial_mode == 1:
    layers = 2*args.nlevels
else:
    layers = args.nlevels
use_cuda = False

output_dim_dict = {
    '2class':2,
    '3class':3,
    '4class':4,
    '9class':9
}
criterion_dict = {
    'deap':'CrossEntropyLoss',
    'amigos':'CrossEntropyLoss',
    'private':'CrossEntropyLoss'
}
class_dict = {
    'deap9class':['[1,3)&[1,3)','[1,3)&[3,6)','[1,3)&[6,9]',
                       '[3,6)&[1,3)','[3,6)&[3,6)','[3,6)&[6,9]',
                       '[6,9]&[1,3)','[6,9]&[3,6)','[6,9]&[6,9]'],
    'amigos4class': ['LVLA','LVHA','HVLA','HVHA'],
    'amigos9class':['[1,3)&[1,3)','[1,3)&[3,6)','[1,3)&[6,9]',
                       '[3,6)&[1,3)','[3,6)&[3,6)','[3,6)&[6,9]',
                       '[6,9]&[1,3)','[6,9]&[3,6)','[6,9]&[6,9]'],
    'amigoslong4class':['LVLA','LVHA','HVLA','HVHA'],
    'private4class':['moved', 'nervous', 'angry', 'happy'],
    'deap2class':['LOW','HIGH'],
    'deap3class':['[1,3)','[3,6)','[6,9]']
}
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True
hyp_params = args
hyp_params.dataset = dataset[:-1]
hyp_params.data_folder = os.path.join(hyp_params.data_path, dataset)

print("Start loading the data....")
print(dataset)
train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
physio,vision1,vision2,labels = train_data.physio, train_data.vision1, train_data.vision2, train_data.labels
print(physio.shape)
print(vision1.shape)
print(vision2.shape)
print(labels.shape)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

hyp_params.orig_d_p, hyp_params.orig_d_v2, hyp_params.orig_d_v1 = train_data.get_dim()
hyp_params.p_len, hyp_params.v2_len, hyp_params.v1_len = train_data.get_seq_len()
hyp_params.layers = layers
hyp_params.use_cuda = use_cuda

hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.index = dataset[len(dataset)-1]

for key in output_dim_dict:
    if key in dataset:
        hyp_params.output_dim = output_dim_dict.get(key, 1)
        break

for key in criterion_dict:
    if key in dataset:
        hyp_params.criterion = criterion_dict.get(key, 'CrossEntropyLoss')
        break

for key in class_dict:
    if key in dataset:
        hyp_params.class_name = class_dict.get(key,'deap9class0')
        break

mode =''
if args.ponly:
    mode = mode + 'p'
if args.v1only:
    mode = mode + 'v1'
if args.v2only:
    mode = mode + 'v2'
hyp_params.mode = mode
if __name__ == '__main__':
    test_loss = ftrain.initiate(hyp_params, train_loader, valid_loader, test_loader)
