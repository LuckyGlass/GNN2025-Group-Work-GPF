import argparse

from loader import MoleculeDataset
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from splitters import scaffold_split, random_scaffold_split, random_split
import pandas as pd
import graph_prompt as Prompt
import os
import shutil
import random


criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer, prompt):
    model.train()
    all_losses = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # 传入 prompt_type 参数
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt, args.tuning_type)
        y = batch.y.view(pred.shape).to(torch.float64)

        is_valid = y**2 > 0
        loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        all_losses.append(loss.item())

        optimizer.step()
    return np.mean(all_losses)


def eval(args, model, device, loader, prompt):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            # 传入 prompt_type 参数
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, prompt, args.tuning_type)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)


def plot_curves(val_acc_list, test_acc_list, train_loss_list, output_file):
    import matplotlib.pyplot as plt

    epochs = range(1, len(val_acc_list) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    axes[0].plot(epochs, val_acc_list, label='Validation Accuracy')
    axes[0].plot(epochs, test_acc_list, label='Test Accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Curve')
    axes[0].legend()
    
    axes[1].plot(epochs, train_loss_list, label='Train Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Curve')
    axes[1].legend()

    plt.savefig(output_file)
    plt.close()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_scale', type=float, default=1)
    parser.add_argument('--use_cosine', action='store_true', help='Whether to use cosine annealing scheduler.')
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--num_layer', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=300)
    parser.add_argument('--dropout_ratio', type=float, default=0.5)
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--JK', type=str, default="last")
    parser.add_argument('--gnn_type', type=str, default="gin")
    # 修改：添加新的 tuning_type 选项
    parser.add_argument('--tuning_type', type=str, default="gpf", 
                        choices=['gpf', 'gpf-plus', 'gpf_multi', 'gpf_multi_shared'],
                        help='gpf: 原始GPF, gpf-plus: GPF-plus, gpf_multi: 每层不同prompt, gpf_multi_shared: 每层相同prompt')
    parser.add_argument('--dataset', type=str, default='tox21')
    parser.add_argument('--model_file', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=0)
    parser.add_argument('--split', type=str, default="scaffold")
    parser.add_argument('--eval_train', type=int, default=0)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--pnum', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_file', type=str, default="result.log")
    parser.add_argument('--plot_curves', action='store_true', help='Whether to plot training curves')
    args = parser.parse_args()

    os.environ["PYTHONHASHSEED"] = str(args.runseed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # 数据集任务数
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        raise ValueError("Invalid dataset name.")

    # 加载数据集
    def load_dataset():
        dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)
        print(dataset)
        if args.split == "scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
            print("scaffold")
        elif args.split == "random":
            train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
            print("random")
        elif args.split == "random_scaffold":
            smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
            train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed)
            print("random scaffold")
        else:
            raise ValueError("Invalid split option.")
        print(train_dataset[0])
        return train_dataset, valid_dataset, test_dataset

    try:
        train_dataset, valid_dataset, test_dataset = load_dataset()
    except RuntimeError:
        print("Try to remove the `processed` directory and try again.")
        if os.path.exists("dataset/" + args.dataset + "/processed"):
            shutil.rmtree("dataset/" + args.dataset + "/processed")
        train_dataset, valid_dataset, test_dataset = load_dataset()

    tgenerator = torch.Generator()
    tgenerator.manual_seed(args.runseed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=tgenerator)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 设置模型
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK=args.JK, drop_ratio=args.dropout_ratio, graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, head_layer=args.num_layers)
    if not args.model_file == "":
        model.from_pretrained(args.model_file)
    print(model)
    model.to(device)

    # === 根据 tuning_type 创建不同的 prompt ===
    if args.tuning_type == 'gpf':
        print("Using GPF (input layer only)")
        prompt = Prompt.SimplePrompt(args.emb_dim).to(device)
    elif args.tuning_type == 'gpf-plus':
        print("Using GPF-plus (input layer with attention)")
        prompt = Prompt.GPFplusAtt(args.emb_dim, args.pnum).to(device)
    elif args.tuning_type == 'gpf_multi':
        print("Using GPF Multi-Layer (different prompt per layer)")
        prompt = Prompt.GPFMultiLayer(args.num_layer, args.emb_dim).to(device)
    elif args.tuning_type == 'gpf_multi_shared':
        print("Using GPF Multi-Layer Shared (same prompt per layer)")
        prompt = Prompt.GPFMultiLayerShared(args.num_layer, args.emb_dim).to(device)

    # 设置优化器
    model_param_group = []
    model_param_group.append({"params": prompt.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr": args.lr * args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr": args.lr * args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay, amsgrad=False)
    print(optimizer)
    if args.use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    train_loss_list = []

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
        
        train_loss = train(args, model, device, train_loader, optimizer, prompt)
        train_loss_list.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader, prompt)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader, prompt)
        test_acc = eval(args, model, device, test_loader, prompt)

        if args.tuning_type == "gpf_multi":
            with torch.no_grad():
                print(f"train: {train_acc}; val: {val_acc}; test: {test_acc}; loss: {train_loss:.3f}; norm: {[torch.norm(p.data).item() for p in prompt.prompts]}")
        else:
            print("train: %f val: %f test: %f loss: %.3f" % (train_acc, val_acc, test_acc, train_loss))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        print("")
    
    if args.plot_curves:
        base_dir = os.path.dirname(args.output_file)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_curves(val_acc_list, test_acc_list, train_loss_list, os.path.join(base_dir, f"{args.gnn_type}_{args.tuning_type}_{args.dataset}_{timestamp}.png"))

    with open(args.output_file, 'a+') as f:
        f.write(os.path.basename(args.model_file).split('.')[0] + ' ' + args.tuning_type + ' ' + args.dataset + ' ' + str(args.runseed) + ' ' + str(np.array(test_acc_list)[-1]))
        f.write('\n')


if __name__ == "__main__":
    main()
