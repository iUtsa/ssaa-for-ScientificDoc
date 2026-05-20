import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from .ssaa import SemanticStructuralAdaptiveAttention
from .rsape import ResearchStructureAwarePositionalEncoding

@dataclass
class NOVAConfig:
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_dim: int = None
    max_seq_len: int = 2048
    dropout: float = 0.1
    window_size: int = 128
    num_global_tokens: int = 4
    num_learned_connections: int = 32
    num_section_types: int = 11
    max_hierarchy_depth: int = 4
    num_citation_buckets: int = 6
    rope_ratio: float = 0.5
    num_stages: int = 6
    num_risk_levels: int = 4
    num_quality_levels: int = 5
    def __post_init__(self):
        if self.intermediate_dim is None: self.intermediate_dim = 4 * self.hidden_dim
    @classmethod
    def tiny(cls): return cls(hidden_dim=256, num_layers=4, num_heads=4, num_kv_heads=2, max_seq_len=512)
    @classmethod
    def small(cls): return cls(hidden_dim=512, num_layers=8, num_heads=8, num_kv_heads=4, max_seq_len=2048)
    @classmethod
    def medium(cls): return cls(hidden_dim=896, num_layers=14, num_heads=14, num_kv_heads=7, max_seq_len=2048)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(dim))
    def forward(self, x): return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.w1, self.w2, self.w3 = nn.Linear(h,i,bias=False), nn.Linear(i,h,bias=False), nn.Linear(h,i,bias=False)
    def forward(self, x): return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, config, idx):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(config.hidden_dim), RMSNorm(config.hidden_dim)
        self.attention = SemanticStructuralAdaptiveAttention(config.hidden_dim, config.num_heads, config.num_kv_heads, config.window_size, config.num_global_tokens, config.num_learned_connections, config.dropout)
        self.ffn, self.drop = SwiGLU(config.hidden_dim, config.intermediate_dim), nn.Dropout(config.dropout)
    def forward(self, x, si, mask=None, cache=False, past=None):
        a, kv = self.attention(self.attn_norm(x), si, mask, cache, past)
        x = x + self.drop(a)
        return x + self.drop(self.ffn(self.ffn_norm(x))), kv

class NOVASLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.pos_enc = ResearchStructureAwarePositionalEncoding(config.hidden_dim, config.max_seq_len, config.num_section_types, config.max_hierarchy_depth, config.num_citation_buckets, config.rope_ratio)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.stage_head = nn.Linear(config.hidden_dim, config.num_stages)
        self.risk_head = nn.Linear(config.hidden_dim, config.num_risk_levels)
        self.quality_head = nn.Linear(config.hidden_dim, config.num_quality_levels)
        self.lm_head.weight = self.tok_emb.weight
        self.drop = nn.Dropout(config.dropout)
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02); m.bias is not None and nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
    def forward(self, input_ids, structure_info=None, attention_mask=None, use_cache=False, past_kv=None):
        B, T = input_ids.shape
        dev = input_ids.device
        if structure_info is None:
            structure_info = {'section_ids': torch.zeros(B,T,dtype=torch.long,device=dev), 'hierarchy_positions': torch.zeros(B,T,4,device=dev), 'citation_distances': torch.full((B,T),5,dtype=torch.long,device=dev), 'boundary_flags': torch.zeros(B,T,3,device=dev)}
        h = self.drop(self.pos_enc(self.tok_emb(input_ids), torch.arange(T,device=dev), structure_info))
        if attention_mask is None: attention_mask = torch.triu(torch.full((T,T),float('-inf'),device=dev),1).unsqueeze(0).unsqueeze(0)
        kvs = []
        for i, layer in enumerate(self.layers):
            h, kv = layer(h, structure_info, attention_mask, use_cache, past_kv[i] if past_kv else None)
            if use_cache: kvs.append(kv)
        h = self.norm(h)
        out = {'logits': self.lm_head(h), 'stage_logits': self.stage_head(h[:,-1]), 'risk_logits': self.risk_head(h[:,-1]), 'quality_logits': self.quality_head(h[:,-1]), 'hidden_states': h}
        if use_cache: out['past_kv'] = kvs
        return out
    def compute_loss(self, input_ids, structure_info=None, stage_labels=None, risk_labels=None, quality_labels=None):
        o = self.forward(input_ids, structure_info)
        loss = F.cross_entropy(o['logits'][:,:-1].reshape(-1,self.config.vocab_size), input_ids[:,1:].reshape(-1), ignore_index=0)
        if stage_labels is not None: loss = loss + 0.1*F.cross_entropy(o['stage_logits'], stage_labels)
        if risk_labels is not None: loss = loss + 0.1*F.cross_entropy(o['risk_logits'], risk_labels)
        if quality_labels is not None: loss = loss + 0.1*F.cross_entropy(o['quality_logits'], quality_labels)
        return {'total_loss': loss, 'lm_loss': loss}
    @torch.no_grad()
    def generate(self, ids, max_new=100, temp=1.0, top_k=50):
        self.eval()
        for _ in range(max_new):
            logits = self.forward(ids[:,-self.config.max_seq_len:])['logits'][:,-1]/temp
            if top_k>0: logits[logits<torch.topk(logits,top_k)[0][...,-1,None]]=float('-inf')
            ids = torch.cat([ids, torch.multinomial(F.softmax(logits,-1),1)],1)
        return ids
    def num_parameters(self): return sum(p.numel() for p in self.parameters())
    def save(self, path): torch.save({'config':self.config,'state_dict':self.state_dict()}, path)
    @classmethod
    def load(cls, path, device='cuda'):
        c = torch.load(path, map_location=device)
        m = cls(c['config']); m.load_state_dict(c['state_dict']); return m.to(device)
