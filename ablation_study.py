import os, sys, json, time, argparse, math, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List

@dataclass
class AblationConfig:
    vocab_size: int = 32000
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_dim: int = 1024
    max_seq_len: int = 512
    dropout: float = 0.1
    window_size: int = 128
    num_global_tokens: int = 4
    num_learned_connections: int = 32
    num_section_types: int = 11
    num_citation_buckets: int = 6
    epochs: int = 15
    batch_size: int = 8
    lr: float = 1e-4
    warmup_steps: int = 500
    eval_every: int = 200
    log_every: int = 50

    @classmethod
    def medium(cls):
        return cls(hidden_dim=896, num_layers=14, num_heads=14, num_kv_heads=7,
                   intermediate_dim=3584, max_seq_len=512, batch_size=2,
                   window_size=128, num_global_tokens=4, num_learned_connections=32,
                   eval_every=100, log_every=25, warmup_steps=300, epochs=10)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight

class SwiGLU(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self._build_cache(max_seq_len)
    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cache', emb.cos())
        self.register_buffer('sin_cache', emb.sin())
    def forward(self, seq_len):
        if seq_len > self.cos_cache.size(0):
            self._build_cache(seq_len)
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]

def apply_rotary(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class DenseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, structure_info=None, causal_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S)
        q, k = apply_rotary(q, k, cos, sin)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            scores = scores + causal_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class FixedSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = config.window_size
        self.num_global = config.num_global_tokens
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
    def _fixed_mask(self, seq_len, device):
        mask = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2)
            mask[i, start:end] = 1
        mask[:self.num_global, :] = 1
        mask[:, :self.num_global] = 1
        return mask
    def forward(self, x, structure_info=None, causal_mask=None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S)
        q, k = apply_rotary(q, k, cos, sin)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        sparse_mask = self._fixed_mask(S, x.device).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        if causal_mask is not None:
            scores = scores + causal_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class ContentSparseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = config.window_size
        self.num_global = config.num_global_tokens
        self.num_learned = config.num_learned_connections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.importance_scorer = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 4), nn.GELU(), nn.Linear(config.hidden_dim // 4, 1))
        self.compat_q = nn.Linear(config.hidden_dim, config.hidden_dim // 4)
        self.compat_k = nn.Linear(config.hidden_dim, config.hidden_dim // 4)
    def _learned_mask(self, x, device):
        B, S, _ = x.shape
        importance = torch.sigmoid(self.importance_scorer(x).squeeze(-1))
        cq = self.compat_q(x)
        ck = self.compat_k(x)
        compat = torch.sigmoid(torch.matmul(cq, ck.transpose(-2, -1)) / math.sqrt(cq.size(-1)))
        conn_scores = importance.unsqueeze(-1) * importance.unsqueeze(-2) * compat
        k = min(self.num_learned, S)
        _, top_idx = torch.topk(conn_scores, k, dim=-1)
        hard_mask = torch.zeros_like(conn_scores)
        hard_mask.scatter_(-1, top_idx, 1.0)
        soft_mask = torch.softmax(conn_scores, dim=-1)
        learned = hard_mask.detach() + soft_mask - soft_mask.detach()
        fixed = torch.zeros(S, S, device=device)
        for i in range(S):
            start = max(0, i - self.window_size // 2)
            end = min(S, i + self.window_size // 2)
            fixed[i, start:end] = 1
        fixed[:self.num_global, :] = 1
        fixed[:, :self.num_global] = 1
        return torch.clamp(fixed.unsqueeze(0) + learned, 0, 1)
    def forward(self, x, structure_info=None, causal_mask=None):
        B, S, _ = x.shape
        sparse_mask = self._learned_mask(x, x.device)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S)
        q, k = apply_rotary(q, k, cos, sin)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(sparse_mask.unsqueeze(1) == 0, float('-inf'))
        if causal_mask is not None:
            scores = scores + causal_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class SSAAFullAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = config.window_size
        self.num_global = config.num_global_tokens
        self.num_learned = config.num_learned_connections
        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len)
        self.dropout = nn.Dropout(config.dropout)
        self.content_scorer = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim // 4), nn.GELU(), nn.Linear(config.hidden_dim // 4, 1))
        self.section_importance = nn.Parameter(torch.zeros(config.num_section_types))
        self.boundary_bonus = nn.Parameter(torch.tensor(0.5))
        self.compat_q = nn.Linear(config.hidden_dim, config.hidden_dim // 4)
        self.compat_k = nn.Linear(config.hidden_dim, config.hidden_dim // 4)
        self.section_compat = nn.Parameter(torch.randn(config.num_section_types, config.num_section_types) * 0.1)
        self.citation_weight = nn.Parameter(torch.tensor(0.3))
    def _structural_mask(self, x, structure_info, device):
        B, S, _ = x.shape
        content_imp = self.content_scorer(x).squeeze(-1)
        section_ids = structure_info.get(chr(39)+"section_ids"+chr(39), torch.zeros(B, S, dtype=torch.long, device=device))
        section_imp = F.embedding(section_ids, self.section_importance.unsqueeze(1)).squeeze(-1)
        boundary_flags = structure_info.get(chr(39)+"boundary_flags"+chr(39), torch.zeros(B, S, 3, device=device))
        boundary_imp = boundary_flags[:, :, 0].float() * self.boundary_bonus
        importance = torch.sigmoid(content_imp + section_imp + boundary_imp)
        cq = self.compat_q(x)
        ck = self.compat_k(x)
        content_compat = torch.matmul(cq, ck.transpose(-2, -1)) / math.sqrt(cq.size(-1))
        section_compat = self.section_compat[section_ids.unsqueeze(-1), section_ids.unsqueeze(-2)]
        cite_dist = structure_info.get(chr(39)+"citation_distances"+chr(39), torch.full((B, S), 5, dtype=torch.long, device=device)).float()
        cite_bonus = (torch.exp(-cite_dist.unsqueeze(-1) / 5) * torch.exp(-cite_dist.unsqueeze(-2) / 5) * self.citation_weight)
        compatibility = torch.sigmoid(content_compat + section_compat + cite_bonus)
        conn_scores = importance.unsqueeze(-1) * importance.unsqueeze(-2) * compatibility
        k = min(self.num_learned, S)
        _, top_idx = torch.topk(conn_scores, k, dim=-1)
        hard_mask = torch.zeros_like(conn_scores)
        hard_mask.scatter_(-1, top_idx, 1.0)
        soft_mask = torch.softmax(conn_scores, dim=-1)
        learned = hard_mask.detach() + soft_mask - soft_mask.detach()
        fixed = torch.zeros(S, S, device=device)
        for i in range(S):
            start = max(0, i - self.window_size // 2)
            end = min(S, i + self.window_size // 2)
            fixed[i, start:end] = 1
        fixed[:self.num_global, :] = 1
        fixed[:, :self.num_global] = 1
        return torch.clamp(fixed.unsqueeze(0) + learned, 0, 1)
    def forward(self, x, structure_info=None, causal_mask=None):
        B, S, _ = x.shape
        if structure_info is None:
            structure_info = {}
        sparse_mask = self._structural_mask(x, structure_info, x.device)
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(S)
        q, k = apply_rotary(q, k, cos, sin)
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(sparse_mask.unsqueeze(1) == 0, float('-inf'))
        if causal_mask is not None:
            scores = scores + causal_mask
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

VARIANT_MAP = {'M1': DenseAttention, 'M2': FixedSparseAttention, 'M3': ContentSparseAttention, 'M4': SSAAFullAttention}

class AblationTransformerBlock(nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim)
        self.ffn_norm = RMSNorm(config.hidden_dim)
        self.attention = VARIANT_MAP[variant](config)
        self.ffn = SwiGLU(config.hidden_dim, config.intermediate_dim)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x, structure_info=None, causal_mask=None):
        x = x + self.dropout(self.attention(self.attn_norm(x), structure_info, causal_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x

class AblationModel(nn.Module):
    def __init__(self, config, variant='M4'):
        super().__init__()
        self.config = config
        self.variant = variant
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([AblationTransformerBlock(config, variant) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.vocab_size, config.hidden_dim, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, structure_info=None):
        B, S = input_ids.shape
        device = input_ids.device
        h = self.token_emb(input_ids)
        h = self.dropout(h)
        causal = torch.triu(torch.full((S, S), float('-inf'), device=device), diagonal=1)
        causal = causal.unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            h = layer(h, structure_info, causal)
        h = self.norm(h)
        return self.lm_head(h)
    def compute_loss(self, input_ids, structure_info=None):
        logits = self.forward(input_ids, structure_info)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1), ignore_index=0)
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

class AblationDataset(Dataset):
    def __init__(self, data_path, seq_len=512):
        self.seq_len = seq_len
        data = torch.load(data_path, weights_only=False)
        self.input_ids = data['input_ids']
        self.structure_info = data.get('structure_info', None)
        print(f'  Loaded {len(self.input_ids)} samples from {data_path}')
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        if len(ids) < self.seq_len:
            ids = torch.cat([ids, torch.zeros(self.seq_len - len(ids), dtype=torch.long)])
        ids = ids[:self.seq_len]
        item = {'input_ids': ids}
        if self.structure_info is not None:
            for key in ['section_ids', 'citation_distances', 'boundary_flags', 'hierarchy_positions']:
                if key in self.structure_info:
                    val = self.structure_info[key][idx]
                    if key in ('boundary_flags', 'hierarchy_positions'):
                        d = 3 if key == 'boundary_flags' else 4
                        if len(val) < self.seq_len:
                            val = torch.cat([val, torch.zeros(self.seq_len - len(val), d)])
                        item[key] = val[:self.seq_len]
                    else:
                        fill = 5 if key == 'citation_distances' else 0
                        if len(val) < self.seq_len:
                            val = torch.cat([val, torch.full((self.seq_len - len(val),), fill, dtype=torch.long)])
                        item[key] = val[:self.seq_len]
        return item

def collate_fn(batch):
    result = {'input_ids': torch.stack([b['input_ids'] for b in batch])}
    struct_keys = ['section_ids', 'citation_distances', 'boundary_flags', 'hierarchy_positions']
    structure_info = {}
    for key in struct_keys:
        if key in batch[0]:
            structure_info[key] = torch.stack([b[key] for b in batch])
    if structure_info:
        result['structure_info'] = structure_info
    return result

def prepare_data(args):
    print('=' * 60)
    print('STEP 1: Preparing Data')
    print('=' * 60)
    if os.path.exists(args.tokenizer):
        with open(args.tokenizer, 'rb') as f:
            tokenizer = pickle.load(f)
        if isinstance(tokenizer, dict):
            from src.tokenizer import BPETokenizer
            tokenizer = BPETokenizer(vocab=tokenizer.get('vocab'), merges=tokenizer.get('merges'))
        vocab_size = len(tokenizer)
        print(f'Loaded tokenizer: vocab_size={vocab_size}')
    else:
        print(f'ERROR: Tokenizer not found at {args.tokenizer}')
        sys.exit(1)
    if not os.path.exists(args.corpus):
        print(f'ERROR: Corpus not found at {args.corpus}')
        sys.exit(1)
    with open(args.corpus, 'r', encoding='utf-8') as f:
        texts = f.readlines()
    print(f'Loaded {len(texts)} documents')
    import random
    random.seed(42)
    random.shuffle(texts)
    max_docs = min(len(texts), 12000)
    texts = texts[:max_docs]
    print(f'Using {len(texts)} documents')
    n_train = int(len(texts) * 0.85)
    n_val = int(len(texts) * 0.075)
    splits = {'train': texts[:n_train], 'val': texts[n_train:n_train+n_val], 'test': texts[n_train+n_val:]}
    print(f'+Split: {len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test')
    os.makedirs('ablation_data', exist_ok=True)
    for split_name, split_texts in splits.items():
        print(f'Tokenizing {split_name}...')
        all_ids, all_sec, all_cite, all_bound, all_hier = [], [], [], [], []
        for i, text in enumerate(split_texts):
            if i % 500 == 0:
                print(f'  {i}/{len(split_texts)}')
            encoded = tokenizer.encode(text, max_length=args.seq_len, padding=True, return_structure=True)
            all_ids.append(torch.tensor(encoded['input_ids'], dtype=torch.long))
            if 'section_ids' in encoded:
                all_sec.append(torch.tensor(encoded['section_ids'], dtype=torch.long))
                all_cite.append(torch.tensor(encoded['citation_distances'], dtype=torch.long))
                all_bound.append(torch.tensor(encoded['boundary_flags'], dtype=torch.float32))
                all_hier.append(torch.tensor(encoded['hierarchy_positions'], dtype=torch.float32))
        save_data = {'input_ids': all_ids}
        if all_sec:
            save_data['structure_info'] = {'section_ids': all_sec, 'citation_distances': all_cite, 'boundary_flags': all_bound, 'hierarchy_positions': all_hier}
        path = f'ablation_data/{split_name}.pt'
        torch.save(save_data, path)
        print(f'  Saved {path} ({len(all_ids)} samples)')
    with open('ablation_data/meta.json', 'w') as f:
        json.dump({'vocab_size': vocab_size, 'seq_len': args.seq_len}, f)
    print('Data preparation complete!')

def train_variant(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 60)
    print(f'Training: {args.variant} | Seed: {args.seed}')
    print('=' * 60)
    print(f'Device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    with open('ablation_data/meta.json', 'r') as f:
        meta = json.load(f)
    if hasattr(args, 'config') and args.config == 'medium':
        config = AblationConfig.medium()
        print('Using MEDIUM config (206M params)')
    else:
        config = AblationConfig()
        print('Using TINY config (12M params)')
    config.vocab_size = meta['vocab_size']
    config.max_seq_len = meta['seq_len']
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    model = AblationModel(config, variant=args.variant).to(device)
    print(f'Parameters: {model.num_parameters():,}')
    train_ds = AblationDataset('ablation_data/train.pt', config.max_seq_len)
    val_ds = AblationDataset('ablation_data/val.pt', config.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    total_steps = config.epochs * len(train_loader)
    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(config.warmup_steps, 1)
        progress = (step - config.warmup_steps) / max(total_steps - config.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    out_dir = f'ablation_results/{args.variant}_seed{args.seed}'
    os.makedirs(out_dir, exist_ok=True)
    log = {'train_loss': [], 'val_loss': [], 'val_ppl': [], 'steps': [], 'val_steps': []}
    global_step = 0
    best_val_loss = float('inf')
    start_time = time.time()
    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            structure_info = None
            if 'structure_info' in batch:
                structure_info = {k: v.to(device) for k, v in batch['structure_info'].items()}
            loss = model.compute_loss(input_ids, structure_info)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()
            log['train_loss'].append(loss.item())
            log['steps'].append(global_step)
            if global_step % config.log_every == 0:
                elapsed = time.time() - start_time
                speed = (global_step * config.batch_size) / elapsed
                print(f'  [{args.variant}] Step {global_step:5d} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | {speed:.0f} tok/s')
            if global_step % config.eval_every == 0:
                model.eval()
                val_total, val_count = 0, 0
                with torch.no_grad():
                    for vb in val_loader:
                        vi = vb['input_ids'].to(device)
                        vs = {k: v.to(device) for k, v in vb['structure_info'].items()} if 'structure_info' in vb else None
                        val_total += model.compute_loss(vi, vs).item()
                        val_count += 1
                avg_val = val_total / max(val_count, 1)
                val_ppl = math.exp(min(avg_val, 20))
                log['val_loss'].append(avg_val)
                log['val_ppl'].append(val_ppl)
                log['val_steps'].append(global_step)
                print(f'  [{args.variant}] VAL Step {global_step} | Loss: {avg_val:.4f} | PPL: {val_ppl:.2f}')
                if avg_val < best_val_loss:
                    best_val_loss = avg_val
                    torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))
                model.train()
        print(f'  [{args.variant}] Epoch {epoch+1}/{config.epochs} | Avg Loss: {epoch_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), os.path.join(out_dir, 'final.pt'))
    total_time = time.time() - start_time
    log.update({'total_time': total_time, 'best_val_loss': best_val_loss,
        'best_val_ppl': math.exp(min(best_val_loss, 20)),
        'num_params': model.num_parameters(), 'variant': args.variant, 'seed': args.seed})
    with open(os.path.join(out_dir, 'log.json'), 'w') as f:
        json.dump(log, f, indent=2)
    print('=' * 60)
    print(f'Done: {args.variant} seed={args.seed} | Time: {total_time/60:.1f}min | Best PPL: {log['best_val_ppl']:.2f}')
    print('=' * 60)

def evaluate_all(args):
    print('=' * 60)
    print('Evaluating All Variants')
    print('=' * 60)
    results = {}
    for variant in ['M1', 'M2', 'M3', 'M4']:
        results[variant] = []
        for seed in [42, 123, 7]:
            path = f'ablation_results/{variant}_seed{seed}/log.json'
            if os.path.exists(path):
                with open(path) as f:
                    results[variant].append(json.load(f))
    print(f'{chr(34)}{chr(86)}{chr(97)}{chr(114)}{chr(105)}{chr(97)}{chr(110)}{chr(116)}{chr(34):<16} {chr(34)}{chr(80)}{chr(97)}{chr(114)}{chr(97)}{chr(109)}{chr(115)}{chr(34):>10} {chr(34)}{chr(86)}{chr(97)}{chr(108)}{chr(32)}{chr(76)}{chr(111)}{chr(115)}{chr(115)}{chr(34):>15} {chr(34)}{chr(86)}{chr(97)}{chr(108)}{chr(32)}{chr(80)}{chr(80)}{chr(76)}{chr(34):>15} {chr(34)}{chr(84)}{chr(105)}{chr(109)}{chr(101)}{chr(34):>10}')
    print('-' * 70)
    names = {'M1': 'Dense', 'M2': 'Fixed', 'M3': 'Content', 'M4': 'SSAA'}
    for v in ['M1', 'M2', 'M3', 'M4']:
        if results[v]:
            losses = [r['best_val_loss'] for r in results[v]]
            ppls = [r['best_val_ppl'] for r in results[v]]
            ml = sum(losses)/len(losses)
            sl = (sum((x-ml)**2 for x in losses)/max(len(losses)-1,1))**0.5
            mp = sum(ppls)/len(ppls)
            sp = (sum((x-mp)**2 for x in ppls)/max(len(ppls)-1,1))**0.5
            t = sum(r['total_time']/60 for r in results[v])/len(results[v])
            print(f'{v} ({names[v]})  {results[v][0]['num_params']:>10,} {ml:>8.4f}+/-{sl:.4f} {mp:>8.2f}+/-{sp:.2f} {t:>8.1f}m')
    print('=' * 70)

def run_all(args):
    for variant in ['M1', 'M2', 'M3', 'M4']:
        for seed in [42, 123, 7]:
            args.variant = variant
            args.seed = seed
            train_variant(args)
    evaluate_all(args)

def main():
    parser = argparse.ArgumentParser(description='NOVA-SLM Ablation Study')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare')
    p.add_argument('--corpus', required=True)
    p.add_argument('--tokenizer', default='tokenizer_sci.pkl')
    p.add_argument('--seq_len', type=int, default=512)
    t = sub.add_parser('train')
    t.add_argument('--variant', required=True, choices=['M1','M2','M3','M4'])
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--config', type=str, default='tiny', choices=['tiny','medium'])
    t.add_argument('--epochs', type=int, default=15)
    t.add_argument('--batch_size', type=int, default=8)
    t.add_argument('--lr', type=float, default=1e-4)
    sub.add_parser('evaluate')
    r = sub.add_parser('run_all')
    r.add_argument('--epochs', type=int, default=15)
    r.add_argument('--batch_size', type=int, default=8)
    r.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.command == 'prepare': prepare_data(args)
    elif args.command == 'train': train_variant(args)
    elif args.command == 'evaluate': evaluate_all(args)
    elif args.command == 'run_all': run_all(args)
    else: parser.print_help()

if __name__ == '__main__':
    main()