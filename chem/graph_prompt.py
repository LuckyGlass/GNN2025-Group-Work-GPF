import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot

from torch_geometric.nn import GATConv


class SimplePrompt(nn.Module):
    def __init__(self, in_channels: int):
        super(SimplePrompt, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.global_emb)

    def add(self, x: Tensor):
        return x + self.global_emb


class GPFplusAtt(nn.Module):
    def __init__(self, in_channels: int, p_num: int):
        super(GPFplusAtt, self).__init__()
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()

    def add(self, x: Tensor):
        score = self.a(x)
        # weight = torch.exp(score) / torch.sum(torch.exp(score), dim=1).view(-1, 1)
        weight = F.softmax(score, dim=1)
        p = weight.mm(self.p_list)

        return x + p
    

class GPFMultiLayer(nn.Module):
    """每层GNN使用不同的prompt"""
    def __init__(self, num_layers: int, in_channels: int):
        super(GPFMultiLayer, self).__init__()
        self.num_layers = num_layers
        
        # 每层一个独立可学习的 prompt
        self.prompts = nn.ParameterList([
            nn.Parameter(torch.Tensor(1, in_channels))
            for _ in range(num_layers)
        ])
        self.reset_parameters()
    
    def reset_parameters(self):
        for prompt in self.prompts:
            glorot(prompt)
    
    def add(self, x: Tensor, layer_idx: int):
        """在第 layer_idx 层添加对应的 prompt"""
        return x + self.prompts[layer_idx]

# add
class GPFMultiLayerShared(nn.Module):
    """每层GNN使用相同的prompt（消融实验用）"""
    def __init__(self, num_layers: int, in_channels: int):
        super(GPFMultiLayerShared, self).__init__()
        self.num_layers = num_layers
        self.shared_prompt = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.shared_prompt)
    
    def add(self, x: Tensor, layer_idx: int):
        """所有层使用同一个 prompt"""
        return x + self.shared_prompt

