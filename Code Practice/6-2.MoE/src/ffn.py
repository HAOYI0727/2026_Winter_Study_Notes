import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float]):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, ffn_dim_multiplier: Optional[float], num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 门控网络：决定每个 Token 去往哪个专家
        self.gate = nn.Linear(dim, num_experts, bias=False)
        # 专家列表：创建 num_experts 个独立的 FeedForward 网络
        self.experts = nn.ModuleList([
            FeedForward(dim, hidden_dim, multiple_of, ffn_dim_multiplier)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, dim)
        B, T, D = x.shape
        x_flat = x.view(-1, D)
        
        # 1. 门控网络
        gate_logits = self.gate(x_flat) # (B*T, num_experts)
        # 2. Top-k 路由
        weights, indices = torch.topk(gate_logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1) # 归一化权重
        
        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            # 3. 找出所有选中当前专家 i 的 token 索引
            batch_idx, k_idx = torch.where(indices == i)
            
            if len(batch_idx) == 0:
                continue
                
            # 4. 取出对应的输入进行计算
            expert_input = x_flat[batch_idx]
            expert_out = expert(expert_input)
            
            # 5. 获取对应的权重
            expert_weights = weights[batch_idx, k_idx].unsqueeze(-1) # (num_selected, 1)
            
            # 6. 将结果加权累加回输出张量
            output.index_add_(0, batch_idx, expert_out * expert_weights)
            
        return output.view(B, T, D)