import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import sys
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif,init_network

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attack', default='fgm', type=str, choices=['none', 'pgd', 'fgsm','fgm','free'])
    parser.add_argument('--epsilon', default=0.001, type=float)
    parser.add_argument('--alpha', default=0.001, type=float)
    parser.add_argument('--attack-iters', default=40, type=int)
    parser.add_argument('--lr-max', default=5e-3, type=float)
    parser.add_argument('--lr-type', default='cyclic')
    parser.add_argument('--fname', default='adv_model', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='TextCNN',type=str)
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    return parser.parse_args()

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}
        #self.attack=attack
    def attack(self, attack,epsilon=1, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    if attack=="fgm":
                        r_at = epsilon * param.grad / norm
                    elif attack=="fgsm":
                        r_at = epsilon * torch.sign(param.grad)
                    param.data.add_(r_at)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1, alpha=0.3, emb_name='emb', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


def main():
    args = get_args()
    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = 'THUCNews'  # 数据集
    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model      
    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    vocab, train_data, _, _ = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    config.n_vocab = len(vocab)
    
    model = x.Model(config).to(config.device)
    init_network(model)
    model.train()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    if args.lr_type == 'cyclic': 
        lr_schedule = lambda t: np.interp([t], [0, config.num_epochs * 2//5, config.num_epochs], [0, args.lr_max, 0])[0]
    elif args.lr_type == 'flat': 
        lr_schedule = lambda t: args.lr_max
    else:
        raise ValueError('Unknown lr_type')

    criterion = nn.CrossEntropyLoss()

    logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')
    
    
    
    if (args.attack=="fgm" or args.attack=="fgsm"):
        fgm = FGM(model)
        for epoch in range(config.num_epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_iter):
                lr = lr_schedule(epoch + (i+1)/len(train_iter))
                opt.param_groups[0].update(lr=lr)
                
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward(retain_graph=True)
                fgm.attack(args.attack)
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward(retain_graph=True)#梯度会累加，原始没干扰的和有干扰的
                #还原emb参数值
                fgm.restore()
                opt.step()
                opt.zero_grad()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            train_time = time.time()
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
            torch.save(model.state_dict(), args.fname+"_"+args.attack)
    if (args.attack=="pgd"):
        pgd = PGD(model)
        K=3
        for epoch in range(config.num_epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_iter):
                lr = lr_schedule(epoch + (i+1)/len(train_iter))
                opt.param_groups[0].update(lr=lr)
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward(retain_graph=True)
                pgd.backup_grad()
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    output = model(X)
                    loss = F.cross_entropy(output, y)
                    loss.backward(retain_graph=True)
                pgd.restore() # 恢复embedding参数
                # 梯度下降，更新参数
                opt.step()
                opt.zero_grad()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            train_time = time.time()
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
            torch.save(model.state_dict(), args.fname+"_"+args.attack)
    if (args.attack=="free"):
        pgd = PGD(model)
        K=3
        for epoch in range(config.num_epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_iter):
                lr = lr_schedule(epoch + (i+1)/len(train_iter))
                opt.param_groups[0].update(lr=lr)
                
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward(retain_graph=True)
                pgd.backup_grad()
     
                # 对抗训练
                for t in range(K):
                    pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                    if t != K-1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                        pgd.restore()
                    output = model(X)
                    loss = F.cross_entropy(output, y)
                    loss.backward(retain_graph=True)
                    opt.step()
                #pgd.restore() # 恢复embedding参数
                # 梯度下降，更新参数
                #opt.step()
                opt.zero_grad()
                
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)

            train_time = time.time()
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
            torch.save(model.state_dict(), args.fname+"_"+args.attack)

    if (args.attack=="none"):
        for epoch in range(config.num_epochs):
            start_time = time.time()
            train_loss = 0
            train_acc = 0
            train_n = 0
            for i, (X, y) in enumerate(train_iter):
                lr = lr_schedule(epoch + (i+1)/len(train_iter))
                opt.param_groups[0].update(lr=lr)
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward()
                opt.step()
                opt.zero_grad()
                train_loss += loss.item() * y.size(0)
                train_acc += (output.max(1)[1] == y).sum().item()
                train_n += y.size(0)
            train_time = time.time()
            logger.info('%d \t %.1f \t %.4f \t %.4f \t %.4f',
                epoch, train_time - start_time, lr, train_loss/train_n, train_acc/train_n)
            torch.save(model.state_dict(), args.fname+"_"+args.attack)
if __name__ == "__main__":
    main()
