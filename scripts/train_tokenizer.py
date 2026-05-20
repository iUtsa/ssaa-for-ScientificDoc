#!/usr/bin/env python3
"""
NOVA-SLM v2: Train BPE Tokenizer
=================================

Train the BPE tokenizer on the downloaded corpus.

Usage:
    python train_tokenizer.py --corpus data/arxiv_corpus.txt --vocab_size 32000 --output tokenizer.pkl
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description='Train BPE tokenizer')
    parser.add_argument('--corpus', type=str, required=True,
                        help='Path to training corpus')
    parser.add_argument('--vocab_size', type=int, default=32000,
                        help='Vocabulary size')
    parser.add_argument('--output', type=str, default='tokenizer.pkl',
                        help='Output path for tokenizer')
    parser.add_argument('--max_lines', type=int, default=None,
                        help='Maximum lines to use (for testing)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("NOVA-SLM v2: BPE Tokenizer Training")
    print("="*60)
    print(f"Corpus: {args.corpus}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Load corpus
    print("\n[1/3] Loading corpus...")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if args.max_lines:
        lines = lines[:args.max_lines]
    
    print(f"  Loaded {len(lines)} documents")
    print(f"  Total characters: {sum(len(l) for l in lines):,}")
    
    # Create and train tokenizer
    print("\n[2/3] Training tokenizer...")
    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    tokenizer.train(lines, verbose=True)
    
    # Save
    print("\n[3/3] Saving tokenizer...")
    tokenizer.save(args.output)
    
    # Test
    print("\n" + "="*60)
    print("Testing tokenizer:")
    test_text = "<abstract> We present a novel transformer model for NLP."
    encoded = tokenizer.encode(test_text, max_length=32, padding=True)
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"  Input:   {test_text}")
    print(f"  Encoded: {encoded['input_ids'][:15].tolist()}...")
    print(f"  Decoded: {decoded[:50]}...")
    print("="*60)
    print("Done!")


if __name__ == '__main__':
    main()
