import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

class StructuralImportanceScorer(nn.Module):
    def __init__(self, hidden_dim, num_section_types=11):
        super().__init__()
        self.section_importance = nn.Parameter(torch.zeros(num_section_types))
        self.content_scorer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//4), nn.GELU(), nn.Linear(hidden_dim//4, 1))
        self.boundary_bonus = nn.Parameter(torch.tensor(0.5))
    def forward(self, hidden_states, structure_info):
        B, T, _ = hidden_states.shape
        content_scores = self.content_scorer(hidden_states).squeeze(-1)
        section_scores = torch.zeros(B, T, device=hidden_states.device)
        if 'section_ids' in structure_info:
            section_scores = F.embedding(structure_info['section_ids'], self.section_importance.unsqueeze(1)).squeeze(-1)
        boundary_scores = torch.zeros(B, T, device=hidden_states.device)
        if 'boundary_flags' in structure_info:
            boundary_scores = structure_info['boundary_flags'][:,:,0].float() * self.boundary_bonus
        return torch.sigmoid(content_scores + section_scores + boundary_scores)

class StructuralCompatibilityScorer(nn.Module):
    def __init__(self, hidden_dim, num_section_types=11):
        super().__init__()
        self.compatibility = nn.Parameter(torch.randn(num_section_types, num_section_types) * 0.1)
        self.citation_weight = nn.Parameter(torch.tensor(0.3))
        self.query_proj = nn.Linear(hidden_dim, hidden_dim//4)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim//4)
    def forward(self, hidden_states, structure_info):
        B, T, _ = hidden_states.shape
        q, k = self.query_proj(hidden_states), self.key_proj(hidden_states)
        content_compat = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(q.size(-1))
        section_compat = torch.zeros(B, T, T, device=hidden_states.device)
        if 'section_ids' in structure_info:
            ids = structure_info['section_ids']
            section_compat = self.compatibility[ids.unsqueeze(-1), ids.unsqueeze(-2)]
        return torch.sigmoid(content_compat + section_compat)

class DifferentiableTopK(nn.Module):
    def __init__(self, k, temperature=1.0):
        super().__init__()
        self.k = k
        self.temperature = temperature
    def forward(self, scores):
        B, T, _ = scores.shape
        k = min(self.k, T)
        _, top_idx = torch.topk(scores, k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(-1, top_idx, 1.0)
        soft = torch.softmax(scores / self.temperature, dim=-1)
        return mask.detach() + soft - soft.detach()

class SemanticStructuralAdaptiveAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_kv_heads, window_size=128, num_global_tokens=4, num_learned_connections=32, dropout=0.1):
        super().__init__()
        self.hidden_dim, self.num_heads, self.num_kv_heads = hidden_dim, num_heads, num_kv_heads
        self.head_dim = hidden_dim // num_heads
        self.window_size, self.num_global = window_size, num_global_tokens
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.importance_scorer = StructuralImportanceScorer(hidden_dim)
        self.compatibility_scorer = StructuralCompatibilityScorer(hidden_dim)
        self.topk_selector = DifferentiableTopK(num_learned_connections)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states, structure_info, attention_mask=None, use_cache=False, past_kv=None):
        B, T, _ = hidden_states.shape
        device = hidden_states.device
        importance = self.importance_scorer(hidden_states, structure_info)
        compatibility = self.compatibility_scorer(hidden_states, structure_info)
        local_mask = torch.zeros(T, T, device=device)
        for i in range(T):
            start, end = max(0, i - self.window_size//2), min(T, i + self.window_size//2)
            local_mask[i, start:end] = 1
        local_mask = local_mask.unsqueeze(0).expand(B, -1, -1)
        global_mask = torch.zeros(B, T, T, device=device)
        global_mask[:, :self.num_global, :] = 1
        global_mask[:, :, :self.num_global] = 1
        conn_scores = importance.unsqueeze(-1) * importance.unsqueeze(-2) * compatibility
        learned_mask = self.topk_selector(conn_scores)
        combined_mask = torch.clamp(local_mask + global_mask + learned_mask, 0, 1)
        q = self.q_proj(hidden_states).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if self.num_kv_heads < self.num_heads:
            k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
            v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None
        attn = torch.matmul(q, k.transpose(-2,-1)) * self.scale
        attn = attn.masked_fill(combined_mask.unsqueeze(1) == 0, float('-inf'))
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out), new_kv
