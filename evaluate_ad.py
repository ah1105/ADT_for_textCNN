import argparse
import logging
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#f1_score(y_test, y_predict)


logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s %(filename)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=100, type=int)
    parser.add_argument('--data-dir', default='../adv-data', type=str)
    parser.add_argument('--fname', default='./adv_model_base',type=str)
    parser.add_argument('--attack', default='none', type=str, choices=['pgd', 'none'])
    parser.add_argument('--epsilon', default=0.01, type=float)
    parser.add_argument('--attack-iters', default=50, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--restarts', default=10, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model', default='TextCNN',type=str) 
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')    
    return parser.parse_args()

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


def main(fn):
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
    vocab, _, _, test_data = build_dataset(config, args.word)
    #train_iter = build_iterator(train_data, config)
    #dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    checkpoint = torch.load(fn)
    model.load_state_dict(checkpoint)
    model.eval()

    total_loss = 0
    total_acc = 0
    n = 0
    pgd = PGD(model)
    K=3
    ytest=[]
    ypre=[]
    if args.attack == 'none':
        with torch.no_grad():
            for i, (X, y) in enumerate(test_iter):
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                n += y.size(0)
                #print(y)
                #print(output.max(1)[1])
                #sys.exit()
                ytemp_test=y.tolist()
                ytest.extend(ytemp_test)
                ytemp_pre=output.max(1)[1].tolist()
                ypre.extend(ytemp_pre)
                
    else:
        for i, (X, y) in enumerate(test_iter):
            output = model(X)
            loss = F.cross_entropy(output, y)
            loss.backward(retain_graph=True)
            pgd.backup_grad()        
            for t in range(K):
            
                pgd.attack(is_first_attack=(t==0)) 
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                output = model(X)
                loss = F.cross_entropy(output, y)
                loss.backward(retain_graph=True)
               
            with torch.no_grad():
                output = model(X)
                loss = F.cross_entropy(output, y)
                total_loss += loss.item() * y.size(0)
                total_acc += (output.max(1)[1] == y).sum().item()
                
                n += y.size(0)
                ytemp_test=y.tolist()
                ytest.extend(ytemp_test)
                ytemp_pre=output.max(1)[1].tolist()
                ypre.extend(ytemp_pre)
    accuracy=accuracy_score(ytest,ypre)
    precision=precision_score(ytest,ypre,average='macro')
    recall=recall_score(ytest,ypre,average='macro')
    f1=f1_score(ytest,ypre,average='macro')
    
    logger.info('model_type: %s Loss: %.4f, Acc: %.4f,accuracy: %.4f ,precision: %.4f ,recall: %.4f,f1: %.4f ', fn,total_loss/n, total_acc/n,accuracy,precision,recall,f1)
    
    

if __name__ == "__main__":
    #fname=["adv_model_none","adv_model_fgm","adv_model_fgsm","adv_model_pgd","adv_model_free"]
    fname=["adv_model_none0.5","adv_model_fgm0.5","adv_model_fgsm0.5","adv_model_pgd0.5","adv_model_free0.5"]
    
    for fn in fname:
        main(fn)
