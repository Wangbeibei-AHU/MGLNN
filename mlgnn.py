import time
import torch
import torch.nn as nn
from copy import deepcopy
import torch.nn.functional as F
import torch.optim as optim
from model import EstimateAdj
from pgd import PGD, prox_operators


class MLGNN:
    def __init__(self, model, args, device):
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_val_loss = 10
        self.val_loss = 0
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.Sigma = None
        self.S = None
        self.m_adj = None
        self.model = model.to(device)


    def fit(self, features, adj, labels, idx_train, idx_val, **kwargs):
        
        args = self.args

        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
       
        self.m_adj = torch.stack(adj, dim=0).mean(dim=0)
        self.Sigma = 1/len(adj)*torch.ones(1,len(adj)).cuda()
        
        estimator = EstimateAdj(self.m_adj, symmetric=args.symmetric, device=self.device).to(self.device)
        self.estimator = estimator
        self.optimizer_adj = optim.SGD(estimator.parameters(), momentum=0.9, lr=args.lr_adj)
        self.optimizer_l1 = PGD(estimator.parameters(), proxs=[prox_operators.prox_l1], lr=args.lr_l1, alphas=[args.beta])       

        # Train model
        for epoch in range(args.epochs):
            
            for i in range(int(args.outer_steps)):
                self.train_adj(epoch, features, adj, labels, idx_train, idx_val)
                self.Sigma = self.sigma_update(self.S, adj)
                                
            for i in range(int(args.inner_steps)):
                self.train_gcn(epoch, features, estimator.estimated_adj, labels, idx_train, idx_val)
    
        print("Optimization Finished!")
        # Testing
        self.model.load_state_dict(self.weights)
        
    def train_gcn(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator
        adj = estimator.normalize()

        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(features, adj)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = self.accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        # Evaluate validation set performance separately, deactivates dropout during validation run.
        self.model.eval()
        output = self.model(features, adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = self.accuracy(output[idx_val], labels[idx_val])

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = adj.detach()
            self.val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())
            
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = adj.detach()
            self.val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())


    def train_adj(self, epoch, features, adj, labels, idx_train, idx_val):
        args = self.args
        estimator = self.estimator

        t = time.time()
        estimator.train()
        self.optimizer_adj.zero_grad()

        loss_l1 = torch.norm(estimator.estimated_adj, 1)
        loss_fro = 0
        for i in range(len(adj)):
            loss_fro += self.Sigma[0][i] * torch.norm((estimator.estimated_adj - adj[i]), p='fro') 
          
        normalized_adj = estimator.normalize()

        output = self.model(features, normalized_adj)
        loss_gcn = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = self.accuracy(output[idx_train], labels[idx_train])

        loss_diffiential = loss_fro + args.lambda_ * loss_gcn 
        loss_diffiential.backward()
        self.optimizer_adj.step()
        self.optimizer_l1.zero_grad()
        self.optimizer_l1.step()
        
        total_loss = loss_diffiential + args.beta * loss_l1
        
        estimator.estimated_adj.data.copy_(torch.clamp(estimator.estimated_adj.data, min=0, max=1))
        self.S = deepcopy(estimator.estimated_adj.data)

        # Evaluate validation set performance separately, deactivates dropout during validation run.
        self.model.eval()
        normalized_adj = estimator.normalize()
        output = self.model(features, normalized_adj)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = self.accuracy(output[idx_val], labels[idx_val])
        # print('Epoch: {:04d}'.format(epoch+1),
        #       'loss_gcn: {:.4f}'.format(loss_gcn.item()),
        #       'acc_train: {:.4f}'.format(acc_train.item()),
        #       'loss_val: {:.4f}'.format(loss_val.item()),
        #       'acc_val: {:.4f}'.format(acc_val.item()),
        #       'time: {:.4f}s'.format(time.time() - t))

        if acc_val > self.best_val_acc:
            self.best_val_acc = acc_val
            self.best_graph = normalized_adj.detach()
            self.val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())
            
        if loss_val < self.best_val_loss:
            self.best_val_loss = loss_val
            self.best_graph = normalized_adj.detach()
            self.val_loss = loss_val
            self.weights = deepcopy(self.model.state_dict())
            

    def test(self, features, labels, idx_val, idx_test):
        """Evaluate the performance on test set
        """
        print("\t=== testing ===")
        self.model.eval()
        adj = self.best_graph
        if self.best_graph is None:
            adj = self.estimator.normalize()
        output = self.model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = self.accuracy(output[idx_test], labels[idx_test])
               
        print("\tTest set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test.item()


    def sigma_update(self, S, adj):
        v = 1/len(adj)*torch.ones(1,len(adj)).cuda()
        for num in range(len(adj)):
            v[0,num] = -0.5*(torch.norm((S - adj[num]), p='fro')*torch.norm((S - adj[num]), p='fro'))/self.args.gamma
        k = 1
        ft=1
        n = len(adj)
        v0 = v-torch.mean(v) + k/n
        vmin = torch.min(v0)
        posidx = torch.zeros(v0.size()).cuda()
        if vmin < 0:
            f = 1
            lambda_m = 0
            while abs(f) > 1e-5:
                v1 = v0 - lambda_m
                posidx[v1>0] = 1
                posidx[v1<=0] = 0
                npos = torch.sum(posidx)
                g = -npos
                f = torch.matmul(v1, torch.t(posidx))-k
                lambda_m = lambda_m - f/g
                ft=ft+1
                if ft > 100:
                    x = v1
                    x[v1<0] = 0
                    #x[v1>1] = 1
                    break
            x = v1
            x[v1<0] = 0
            #x[v1>1] = 1
        else:
            x = v0
        return x


    def accuracy(self, output, labels):
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

