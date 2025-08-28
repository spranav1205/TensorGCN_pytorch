from __future__ import division, print_function

import argparse
import json
import os
import random
import time

# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import deepdish as dd
from tqdm import trange
import pickle as pkl

from models_pytorch import TGCN
# from torch_geometric.data.sampler import NeighborSampler
from torch_geometric.loader import NeighborSampler
from utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

dataset = 'mr'

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--no_sparse', action='store_true')
    parser.add_argument("--load_ckpt", action='store_true')
    parser.add_argument('--featureless', action='store_true')
    parser.add_argument("--save_path", type=str, default='./saved_model', help="the path of saved model")
    parser.add_argument('--dataset', type=str, default='mr', help='dataset name, default to mr')
    parser.add_argument('--model', type=str, default='gcn', help='model name, default to gcn')
    parser.add_argument('--lr', '--learning_rate', default=0.0005, type=float)   # 0.002/0.0002
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--hidden", default=200, type=int)
    parser.add_argument("--layers", default=2, type=int)
    parser.add_argument("--dropout", default=0.3, type=float)   # 0.5/0.3/0.1
    parser.add_argument("--weight_decay", default=1.25e-05, type=float)
    parser.add_argument("--early_stop", default=2000, type=int)
    parser.add_argument("--max_degree", default=3, type=int)
    parser.add_argument("--mod_loss", action='store_true')
    return parser.parse_args(args)

def save_model(model, optimizer, args):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}
    torch.save(save_dict, os.path.join(args.save_path, 'model.bin'))
    


def train(args, features, modularity_matrix, train_label, train_mask, val_label, val_mask, test_label, test_mask, model, indice_list, weight_list, doc_indices=None, lam = 0.01, gamma = 0.97):
    cost_valid = []
    acc_valid = []
    max_acc = 0.0
    min_cost = 10.0

    best_val = 0.0
    best_epoch = 0
    corr_test = 0.0
    # for (name, param) in model.named_parameters():
    #     print(name)
    # weight_decay_list = (param for (name, param) in model.named_parameters() if 'layers.0' in name)
    # no_decay_list = (param for (name, param) in model.named_parameters() if 'layers.0' not in name)
    # parameters = [{'params':weight_decay_list},{'params':no_decay_list, 'weight_decay':0.0}]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train()
    for epoch in range(args.epochs):
        
        t = time.time()
        # Construct feed dictionary
        # feed_dict = construct_feed_dict(
        #     features, support, support_mix, y_train, train_mask, placeholders)
        # feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        # print("The number of points =", len(train_label))
        outs = model(features, indice_list, weight_list, 1-args.dropout)
        # Replace outputs of train data with the train labels before passing it

        # Slice doc-only labels and masks
        train_label_docs = train_label[doc_indices]        # [num_docs]
        train_mask_docs  = train_mask[doc_indices].bool()  # [num_docs]

        # Softmax over docs
        output_prob = F.softmax(outs[doc_indices], dim=1).clone()

        # One-hot only needs num_classes (2 in your case)
        output_prob[train_mask_docs] = F.one_hot(
            train_label_docs[train_mask_docs],
            num_classes=output_prob.size(1)
        ).float()
      
        if not isinstance(modularity_matrix, torch.Tensor):
            modularity_matrix = torch.tensor(modularity_matrix, dtype=torch.float, device=device)
        else:
            modularity_matrix = modularity_matrix.detach().clone().to(device)

        # Initialize modularity loss
        loss_mod_W = 0.0
        if args.mod_loss:
            # Degree vector and total edge weight
            D_W = torch.sum(modularity_matrix, dim=0)
            e_W = torch.sum(modularity_matrix) / 2.0

            # Modularity matrix
            B_W = modularity_matrix - gamma * torch.outer(D_W, D_W) / (2.0 * e_W)

            # Optional: normalize to prevent huge values
            B_W = B_W / (torch.max(torch.abs(B_W)) + 1e-8)

            # Modularity loss
            loss_mod_W = torch.trace(output_prob.T @ B_W @ output_prob) / (2.0 * e_W)
            print("Modularity loss:", loss_mod_W.item())

        pre_loss = loss_fct(outs, train_label)
        train_pred = torch.argmax(outs, dim=-1)

        ce_loss = (pre_loss * train_mask/train_mask.mean()).mean()

        train_acc = ((train_pred == train_label).float() * train_mask/train_mask.mean()).mean()
        # loss = ce_loss + tmp_loss
        loss = lam * ce_loss + (1 - lam) * loss_mod_W  # Weighted contribution from both modularity and cross-entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        model.eval()
        # Validation
        valid_cost, valid_acc, pred, labels, duration = evaluate(
            features, val_label, val_mask, model, indice_list, weight_list)

        # Testing
        test_cost, test_acc, pred, labels, test_duration = evaluate(
            features, test_label, test_mask, model, indice_list, weight_list)
        
        if valid_acc > best_val:
            best_val = valid_acc
            best_epoch = epoch
            corr_test = test_acc
        
        model.train()
        cost_valid.append(valid_cost)
        acc_valid.append(valid_acc)
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(loss.item()), "train_acc=",
            "{:.5f}".format(train_acc.item()), "val_loss=", "{:.5f}".format(valid_cost),
            "val_acc=", "{:.5f}".format(valid_acc), "test_loss=", "{:.5f}".format(test_cost), "test_acc=",
            "{:.5f}".format(test_acc), "best_val=", "{:.5f}".format(best_val), "corr_test=", "{:.5f}".format(corr_test)) #, "time=", "{:.5f}".format(time.time() - t)
        

        # save model
        if epoch > 700 and cost_valid[-1] < min_cost:
            save_model(model, optimizer, args)
            min_cost = cost_valid[-1]
            print("Current best loss {:.5f}".format(min_cost))

        # if acc_valid[-1] > max_acc:
        #     save_model(model, optimizer, args)
        #     min_cost = cost_valid[-1]
        #     max_acc = acc_valid[-1]
        #     print("Current best acc {:.5f}".format(max_acc))

        if epoch > args.early_stop and cost_valid[-1] > np.mean(cost_valid[-(args.early_stop + 1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

def evaluate(features, label, mask, model, indice_list, weight_list):
    t_test = time.time()
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        outs = model(features, indice_list, weight_list, 1)
        pre_loss = loss_fct(outs, label)
        pred = torch.argmax(outs, dim=-1)
        ce_loss = (pre_loss * mask/mask.mean()).mean()
        loss = ce_loss
        acc = ((pred == label).float() * mask/mask.mean()).mean()
    # feed_dict_val = construct_feed_dict(
    #     features, support, support_mix, labels, mask, placeholders)
    # outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return loss.item(), acc.item(), pred.cpu().numpy(), label.cpu().numpy(), (time.time() - t_test)

def load_ckpt(model):
    model_dict = model.state_dict()
    pretrained_dict = dd.io.load('./gcn.h5')
    model_dict['layers.0.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.0.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.0.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_1_vars_weights_22:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_0:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.0'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_00:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_1:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.1'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_11:0'].T, dtype=torch.float)
    model_dict['layers.1.intra_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_2:0'].T, dtype=torch.float)
    model_dict['layers.1.inter_convs.2'] = torch.tensor(pretrained_dict['gcn_graphconvolution_mix1_2_vars_weights_22:0'].T, dtype=torch.float)    
    model.load_state_dict(model_dict)

# tf.compat.v1.disable_eager_execution()
def get_edge_tensor(adj):
    row = torch.tensor(adj.row, dtype=torch.long)
    col = torch.tensor(adj.col, dtype=torch.long)
    data = torch.tensor(adj.data, dtype=torch.float)
    indice = torch.stack((row,col),dim=0)
    return indice, data


def main(args):
    
    os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.path.abspath(os.path.dirname(os.getcwd()))
    os.path.abspath(os.path.join(os.getcwd(), ".."))
    f_file = os.sep.join(['..', 'data_tgcn', args.dataset, 'build_train'])
    if torch.cuda.is_available():
        device = 'cuda'
    # Set random seed
    #seed = random.randint(1, 200)
    seed=147
    print(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
    # tf.compat.v1.set_random_seed(seed)


    # Load data
    adj, adj1, adj2, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, val_size,test_size, num_labels, doc_indices = load_corpus_torch(args.dataset, device)
    modularity_path = os.path.join(".", "data", f"{dataset}.full.full.modularity_adj")
    with open(modularity_path, "rb") as f:
        modularity_matrix = np.load(f)
        print("Loaded modularity matrix:", modularity_matrix.shape)

    adj = adj.tocoo()
    adj1 = adj1.tocoo()
    adj2 = adj2.tocoo()
    print(adj)

    print(adj.shape)

    # one-hot features
    # features = torch.eye(adj.shape[0], dtype=torch.float).to_sparse().to(device)
    support_mix = [adj, adj1, adj2]
    indice_list, weight_list = [] , []
    for adjacency in support_mix:
        ind, dat = get_edge_tensor(adjacency)
        indice_list.append(ind.to(device))
        weight_list.append(dat.to(device))
        
    in_dim = adj.shape[0]
    model = TGCN(in_dim=in_dim, hidden_dim=args.hidden, out_dim=num_labels,
        num_graphs=3, dropout=args.dropout, n_layers=args.layers, bias=False, featureless=args.featureless)
    features = torch.tensor(list(range(in_dim)), dtype=torch.long).to(device)
    

    model.to(device)
    print("Model #Params: %d" % sum(p.numel() for p in model.parameters()))
    
    if args.do_train:
        print(doc_indices)
        train(args, features, modularity_matrix, y_train, train_mask, y_val, val_mask, y_test, test_mask, model, indice_list, weight_list, doc_indices)

    if args.do_valid:
        # FLAGS.dropout = 1.0
        # save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        # if args.load_ckpt:
        #     load_ckpt(model)
        # else:
        #     model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        val_cost, val_acc, pred, labels, val_duration = evaluate(
            features, y_val, val_mask, model, indice_list, weight_list)
        print("Val set results:", "cost=", "{:.5f}".format(val_cost),
            "accuracy=", "{:.5f}".format(val_acc), "time=", "{:.5f}".format(val_duration))

        val_pred = []
        val_labels = []
        print(len(val_mask))
        for i in range(len(val_mask)):
            if val_mask[i] == 1:
                val_pred.append(pred[i])
                val_labels.append(labels[i])

        print("Val Precision, Recall and F1-Score...")
        print(metrics.classification_report(val_labels, val_pred, digits=4))
        print("Macro average Val Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(val_labels, val_pred, average='macro'))
        print("Micro average Val Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(val_labels, val_pred, average='micro'))

    if args.do_test:
        # FLAGS.dropout = 1.0
        # save_dict = torch.load(os.path.join(args.save_path, 'model.bin'))
        # if args.load_ckpt:
        #     load_ckpt(model)
        # else:
        #     model.load_state_dict(save_dict['model_state_dict'])
        model.eval()
        # Testing
        test_cost, test_acc, pred, labels, test_duration = evaluate(
            features, y_test, test_mask, model, indice_list, weight_list)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

        test_pred = []
        test_labels = []
        print(len(test_mask))
        for i in range(len(test_mask)):
            if test_mask[i] == 1:
                test_pred.append(pred[i])
                test_labels.append(labels[i])

        print("Test Precision, Recall and F1-Score...")
        print(metrics.classification_report(test_labels, test_pred, digits=4))
        print("Macro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
        print("Micro average Test Precision, Recall and F1-Score...")
        print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))

        name = "mod" if args.mod_loss else "base"

        with open("result_"+name+args.dataset+".txt", "w") as f:
            f.write("Test set results: cost= {:.5f}, accuracy= {:.5f}, time= {:.5f}\n".format(test_cost, test_acc, test_duration))
            f.write("Test Precision, Recall and F1-Score:\n")
            f.write(metrics.classification_report(test_labels, test_pred, digits=4))
            f.write("Macro average Test Precision, Recall and F1-Score:\n")
            f.write(str(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro')))
            f.write("\nMicro average Test Precision, Recall and F1-Score:\n")
            f.write(str(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro')))
if __name__ == '__main__':
    main(parse_args())
