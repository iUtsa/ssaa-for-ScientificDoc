"""
NOVA-SLM v2: BPE Tokenizer (From Scratch) - FIXED SPACES
=========================================================
"""
import re
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

@dataclass
class SpecialTokens:
    PAD: str = "<pad>"
    UNK: str = "<unk>"
    BOS: str = "<bos>"
    EOS: str = "<eos>"
    ABSTRACT: str = "<abstract>"
    INTRO: str = "<intro>"
    METHODS: str = "<methods>"
    RESULTS: str = "<results>"
    CONCLUSION: str = "<conclusion>"
    CITE: str = "<cite>"
    
    def get_all(self) -> List[str]:
        return [self.PAD, self.UNK, self.BOS, self.EOS, self.ABSTRACT, 
                self.INTRO, self.METHODS, self.RESULTS, self.CONCLUSION, self.CITE]

SPECIAL_TOKENS = SpecialTokens()

class BPETokenizer:
    def __init__(self, vocab: Optional[Dict] = None, merges: Optional[List] = None, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.special_tokens = SPECIAL_TOKENS
        self.vocab = vocab or {t: i for i, t in enumerate(SPECIAL_TOKENS.get_all())}
        self.merges = merges or []
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {tuple(p): i for i, p in enumerate(self.merges)}
        self.pad_id = self.vocab.get(SPECIAL_TOKENS.PAD, 0)
        self.unk_id = self.vocab.get(SPECIAL_TOKENS.UNK, 1)
        self._special_pattern = re.compile('(' + '|'.join(re.escape(t) for t in SPECIAL_TOKENS.get_all()) + ')')
    
    def train(self, texts: List[str], verbose: bool = True):
        if verbose: print(f"Training BPE on {len(texts)} texts...")
        word_freqs = Counter()
        for text in texts:
            word_freqs.update(re.findall(r'\w+|[^\w\s]', text.lower()))
        
        splits = {w: list(w) for w in word_freqs}
        all_chars = set(c for w in word_freqs for c in w)
        for c in sorted(all_chars):
            if c not in self.vocab: self.vocab[c] = len(self.vocab)
        
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pair_freqs = Counter()
            for word, freq in word_freqs.items():
                split = splits[word]
                for j in range(len(split)-1):
                    pair_freqs[(split[j], split[j+1])] += freq
            if not pair_freqs: break
            best = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best] < 2: break
            
            new_splits = {}
            for word, split in splits.items():
                new_split, j = [], 0
                while j < len(split):
                    if j < len(split)-1 and split[j] == best[0] and split[j+1] == best[1]:
                        new_split.append(best[0] + best[1])
                        j += 2
                    else:
                        new_split.append(split[j])
                        j += 1
                new_splits[word] = new_split
            splits = new_splits
            new_tok = best[0] + best[1]
            self.vocab[new_tok] = len(self.vocab)
            self.merges.append(best)
            if verbose and (i+1) % 1000 == 0: print(f"  Merge {i+1}: {best}")
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {tuple(p): i for i, p in enumerate(self.merges)}
        if verbose: print(f"Final vocab: {len(self.vocab)}")
    
    def _tokenize_word(self, word: str) -> List[str]:
        tokens = list(word)
        while len(tokens) > 1:
            best_pair, best_rank = None, float('inf')
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i+1])
                if pair in self.merge_ranks and self.merge_ranks[pair] < best_rank:
                    best_rank, best_pair = self.merge_ranks[pair], pair
            if best_pair is None: break
            new_tokens, i = [], 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def encode(self, text: str, max_length: int = None, padding: bool = False, return_structure: bool = True) -> Dict:
        parts = self._special_pattern.split(text)
        tokens = [self.special_tokens.BOS]
        for part in parts:
            if not part: continue
            if part.startswith('<'): tokens.append(part)
            else: tokens.extend(w for word in re.findall(r'\w+|[^\w\s]', part.lower()) for w in self._tokenize_word(word))
        tokens.append(self.special_tokens.EOS)
        
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]
        if max_length: ids = ids[:max_length]
        seq_len = len(ids)
        mask = [1] * seq_len
        if padding and max_length:
            ids += [self.pad_id] * (max_length - seq_len)
            mask += [0] * (max_length - seq_len)
            seq_len = max_length
        
        result = {'input_ids': np.array(ids, dtype=np.int32), 'attention_mask': np.array(mask, dtype=np.int32)}
        if return_structure:
            result.update(self._extract_structure(tokens[:seq_len], seq_len))
        return result
    
    def _extract_structure(self, tokens: List[str], seq_len: int) -> Dict:
        section_map = {'<abstract>': 0, '<intro>': 1, '<methods>': 3, '<results>': 5, '<conclusion>': 7}
        section_ids = np.full(seq_len, 10, dtype=np.int32)
        hier = np.zeros((seq_len, 4), dtype=np.int32)
        cite_dist = np.full(seq_len, 5, dtype=np.int32)
        boundary = np.zeros((seq_len, 3), dtype=np.int32)
        
        current_sec, sec_idx, para_idx, sent_idx = 10, 0, 0, 0
        cite_pos = [i for i, t in enumerate(tokens) if t == '<cite>']
        
        for i, tok in enumerate(tokens):
            if tok in section_map:
                current_sec = section_map[tok]
                sec_idx += 1
                boundary[i] = [1, 1, 1]
            section_ids[i] = current_sec
            hier[i] = [sec_idx, para_idx, sent_idx, i % 50]
            if cite_pos:
                d = min(abs(i - c) for c in cite_pos)
                cite_dist[i] = 0 if d == 0 else 1 if d <= 2 else 2 if d <= 5 else 3 if d <= 10 else 4 if d <= 20 else 5
        
        return {'section_ids': section_ids, 'hierarchy_positions': hier, 'citation_distances': cite_dist, 'boundary_flags': boundary}
    
    def decode(self, ids: np.ndarray, skip_special: bool = True) -> str:
        """Decode token IDs back to text WITH SPACES."""
        tokens = []
        for i in ids:
            token = self.id_to_token.get(int(i), '<unk>')
            tokens.append(token)
        
        if skip_special:
            tokens = [t for t in tokens if not (t.startswith('<') and t.endswith('>'))]
        
        # Join with spaces between words
        result = []
        for i, token in enumerate(tokens):
            # Check if token is punctuation
            if token in '.,;:!?()[]{}"\'-':
                result.append(token)
            else:
                if result and result[-1] not in '([{"\'-':
                    result.append(' ')
                result.append(token)
        
        text = ''.join(result).strip()
        
        # Clean up spacing around punctuation
        text = re.sub(r'\s+([.,;:!?\)\]\}])', r'\1', text)
        text = re.sub(r'([(\[\{])\s+', r'\1', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Capitalize after periods
        text = re.sub(r'(\.\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
        
        return text
    
    def __len__(self): return len(self.vocab)
    
    def save(self, path: str):
        with open(path, 'wb') as f: pickle.dump({'vocab': self.vocab, 'merges': self.merges}, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f: d = pickle.load(f)
        return cls(vocab=d['vocab'], merges=d['merges'])

def build_tokenizer(vocab_size: int = 32000, corpus: List[str] = None) -> BPETokenizer:
    if corpus is None:
        corpus = [f"<abstract> Novel method for {t}. <methods> We use transformers. <results> We achieve good results."
                  for t in ['NLP', 'vision', 'ML', 'AI', 'transformers']] * 100
    tok = BPETokenizer(vocab_size=vocab_size)
    tok.train(corpus)
    return tok
