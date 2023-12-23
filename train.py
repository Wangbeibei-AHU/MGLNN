import torch
import argparse
import numpy as np
from model import GCN
from mlgnn import MLGNN
from dataset import get_dataset


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,help='Disables CUDA training.')
parser.add_argument('--dataset', type=str, default='Caltech101_7', choices=['Caltech101_7', 'Caltech101_20', 'Handwritten', 'Reuters', 'CiteSeer', 'WebKB'], help='dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=40, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--view_num', type=int, default=6, help='view_number')
parser.add_argument('--runs', type=int, default=10, help='runs')
parser.add_argument('--train_ratio', type=float, default=.1, help='training label ratio')
parser.add_argument('--valid_ratio', type=float, default=.1, help='validation label ratio')
parser.add_argument('--epochs', type=int,  default=1000, help='Number of epochs to train.')
parser.add_argument('--beta', type=float, default=0.005, help='weight of l1 norm')
parser.add_argument('--lambda_', type=float, default=1., help='weight of gnn loss')
parser.add_argument('--gamma', type=float, default=1e6, help='weight of sigma')
parser.add_argument('--inner_steps', type=int, default=4, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.001, help='lr for training adj')
parser.add_argument('--lr_l1', type=float, default=0.001, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=True, help='whether use symmetric matrix')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print(args) 

results = []
for runs in range(args.runs):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    adj, features, labels, idx_train, idx_val, idx_test = get_dataset(runs, args, device=device)
    
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item() + 1, dropout=args.dropout)
    mlgnn = MLGNN(model, args, device)
    mlgnn.fit(features, adj, labels, idx_train, idx_val)
    results.append(mlgnn.test(features, labels, idx_val, idx_test))

print(args.dataset," ", args.train_ratio, " results: ",  np.mean(results), "+", np.std(results))

