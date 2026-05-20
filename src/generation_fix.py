
import torch
import torch.nn.functional as F
from typing import Optional

def generate_fixed(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_p: float = 0.92,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    min_length: int = 20,
    eos_token_id: int = 3
):

    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device
    
    # Track generated tokens
    generated = input_ids.clone()
    past_ngrams = {}
    
    for step in range(max_new_tokens):
        # Get logits
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs['logits'][:, -1, :]  # [batch, vocab]
        
        # 1. REPETITION PENALTY
        for i in range(batch_size):
            for token_id in generated[i, -50:]:  # Last 50 tokens
                if token_id < logits.shape[-1]:
                    if logits[i, token_id] < 0:
                        logits[i, token_id] *= repetition_penalty
                    else:
                        logits[i, token_id] /= repetition_penalty
        
        # 2. NO REPEAT N-GRAMS
        if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
            for i in range(batch_size):
                # Get last n-1 tokens
                ngram_prefix = tuple(generated[i, -(no_repeat_ngram_size-1):].tolist())
                
                
                gen_tokens = generated[i].tolist()
                for j in range(len(gen_tokens) - no_repeat_ngram_size + 1):
                    ngram = tuple(gen_tokens[j:j+no_repeat_ngram_size-1])
                    if ngram == ngram_prefix:
                        # Ban the next token
                        banned_token = gen_tokens[j + no_repeat_ngram_size - 1]
                        logits[i, banned_token] = -float('inf')
        
        # 3. MIN LENGTH
        if step < min_length:
            logits[:, eos_token_id] = -float('inf')
        
        # 4. TEMPERATURE
        logits = logits / temperature
        
        # 5. TOP-K filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('inf')
        
        # 6. TOP-P (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            for i in range(batch_size):
                indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = -float('inf')
        
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        
        generated = torch.cat([generated, next_token], dim=1)
        
        
        if next_token.item() == eos_token_id:
            break
    
    return generated



def patch_model_generation(model):
    """Replace model.generate with fixed version."""
    original_generate = model.generate
    
    def new_generate(self, input_ids, max_new=150, temp=0.8, top_k=50, top_p=0.92):
        return generate_fixed(
            self, input_ids, 
            max_new_tokens=max_new,
            temperature=temp,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            min_length=20
        )
    
    model.generate = lambda *args, **kwargs: new_generate(model, *args, **kwargs)
    return model

if __name__ == '__main__':
    print("Fixed generation script ready!")
    print("Use: model = patch_model_generation(model)")
