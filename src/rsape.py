"""NOVA-SLM v2: RSAPE - YOUR NOVEL Positional Encoding (PyTorch)"""
import torch
import torch.nn as nn
import math
from typing import Dict

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())
    def forward(self, positions):
        return self.cos_cache[positions], self.sin_cache[positions]

class ResearchStructureAwarePositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_seq_len=4096, num_section_types=11, max_hierarchy_depth=4, num_citation_buckets=6, rope_ratio=0.5):
        super().__init__()
        self.rope_dim = int(hidden_dim * rope_ratio)
        self.struct_dim = hidden_dim - self.rope_dim
        self.rotary_emb = RotaryEmbedding(self.rope_dim, max_seq_len)
        self.section_embedding = nn.Embedding(num_section_types, self.struct_dim // 4)
        self.hierarchy_embedding = nn.Linear(max_hierarchy_depth, self.struct_dim // 4)
        self.citation_embedding = nn.Embedding(num_citation_buckets, self.struct_dim // 4)
        self.boundary_embedding = nn.Linear(3, self.struct_dim // 4)
        self.struct_combiner = nn.Linear(self.struct_dim, self.struct_dim)
        
    def forward(self, hidden_states, positions, structure_info):
        B, T, _ = hidden_states.shape
        device = hidden_states.device
        rope_states = hidden_states[..., :self.rope_dim]
        struct_states = hidden_states[..., self.rope_dim:]
        embeds = []
        if 'section_ids' in structure_info:
            embeds.append(self.section_embedding(structure_info['section_ids'].to(device)))
        else:
            embeds.append(torch.zeros(B, T, self.struct_dim//4, device=device))
        if 'hierarchy_positions' in structure_info:
            h = structure_info['hierarchy_positions'].float().to(device)
            h = h / (h.max(dim=1, keepdim=True)[0] + 1)
            embeds.append(self.hierarchy_embedding(h))
        else:
            embeds.append(torch.zeros(B, T, self.struct_dim//4, device=device))
        if 'citation_distances' in structure_info:
            embeds.append(self.citation_embedding(structure_info['citation_distances'].to(device)))
        else:
            embeds.append(torch.zeros(B, T, self.struct_dim//4, device=device))
        if 'boundary_flags' in structure_info:
            embeds.append(self.boundary_embedding(structure_info['boundary_flags'].float().to(device)))
        else:
            embeds.append(torch.zeros(B, T, self.struct_dim//4, device=device))
        struct_enc = self.struct_combiner(torch.cat(embeds, dim=-1))
        return torch.cat([rope_states, struct_states + struct_enc], dim=-1)
