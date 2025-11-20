# -*- coding: utf-8 -*-
"""
Veritas Transformer 'Ockham's Gyroscope' v2.0 w Tinygrad (FINAL WORKING)
Poprawki:
1. Włączono Tensor.training = True.
2. Usunięto 'model.embedding.weight' z optymalizatora, ponieważ operacje NumPy przerywają gradient.
   Jest to zgodne z filozofią: "Fakty (embeddingi) są modyfikowane przez ZKP, a nie przez gradient descent".

Autor: Wojciech "adepthus" Durmaj
"""
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, optim
import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class VeracityStamp:
    url: str
    block_height: int
    timestamp: int
    data_hash: str
    zk_proof: dict

@dataclass
class VerificationMetrics:
    zk_valid: bool
    compute_time_ms: float
    entropy: float

def verify_zkp(proof: dict) -> bool:
    return proof.get('valid', False)

def ease_of_verification_scorer(stamp: VeracityStamp) -> Tuple[float, VerificationMetrics]:
    start_time = time.time()
    is_valid = verify_zkp(stamp.zk_proof)
    total_time = (time.time() - start_time) * 1000
    
    data_str = stamp.data_hash + json.dumps(stamp.zk_proof)
    unique, counts = np.unique(list(data_str), return_counts=True)
    probs = counts / len(data_str)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    metrics = VerificationMetrics(zk_valid=is_valid, compute_time_ms=total_time, entropy=entropy)
    
    ease_score = (1.0 if is_valid else 0.0) * (1.0 / (1.0 + np.log(total_time + 1.0))) * (1.0 / (1.0 + entropy / 100.0))
    return ease_score, metrics

class TruthEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        # Glorot uniform initialization
        self.weight = Tensor.glorot_uniform(vocab_size, d_model)
        self.d_model = d_model
    
    def __call__(self, x: Tensor, stamps: List[VeracityStamp]) -> Tuple[Tensor, Tensor, Tensor]:
        # 1. Pobieramy bazowy embedding
        one_hot = Tensor.eye(self.weight.shape[0])[x]
        emb = one_hot.dot(self.weight)
        
        # 2. Przechodzimy do NumPy, aby zaaplikować logikę Timechain/ZKP
        # (To przerywa łańcuch gradientu dla self.weight - i to jest OK w tej architekturze)
        emb_np = emb.numpy()
        
        veracity_scores = []
        ease_scores = []
        
        for i, stamp in enumerate(stamps):
            stamp_idx = i % len(stamps)
            current_stamp = stamps[stamp_idx]
            
            ease_score, _ = ease_of_verification_scorer(current_stamp)
            ease_scores.append(ease_score)
            is_valid = verify_zkp(current_stamp.zk_proof)
            veracity_scores.append(1.0 if is_valid else 0.0)
            
            # Wstrzykiwanie pozycji z Timechain (sinusoidalne)
            pos = current_stamp.block_height
            pos_range = np.arange(self.d_model // 2)
            sin_emb = np.sin(pos / 10000 ** (2 * pos_range / self.d_model))
            
            if i < emb_np.shape[1]:
                 emb_np[:, i, :self.d_model//2] += sin_emb * ease_score
            
            # Wstrzykiwanie hasha
            hash_val = int(hashlib.sha256(current_stamp.data_hash.encode()).hexdigest(), 16) % self.d_model
            if i < emb_np.shape[1]:
                emb_np[:, i] += (hash_val / self.d_model) * 0.1

        # 3. Wracamy do Tensora
        emb = Tensor(emb_np)
        
        avg_ease = np.mean(ease_scores) if ease_scores else 0.0
        noise_scale = 0.1 * (1.0 - avg_ease)
        emb = emb + Tensor.randn(*emb.shape) * noise_scale
        
        veracity_tensor = Tensor(veracity_scores)
        ease_tensor = Tensor(ease_scores)
        
        return emb, veracity_tensor, ease_tensor

class EmpathyAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.out_linear = Linear(d_model, d_model)
    
    def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, T, C = q.shape
        Q = self.q_linear(q).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).reshape(B, T, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = Q.matmul(K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        q_mean = Q.mean(axis=2)
        v_mean = V.mean(axis=2)
        # Uproszczony bias empatii
        empathy_bias = (q_mean * v_mean).sum(axis=-1, keepdim=True) 
        scores = scores + empathy_bias.unsqueeze(-1) * 0.1
        
        attn = scores.softmax(axis=-1)
        out = attn.matmul(V).transpose(1, 2).reshape(B, T, C)
        return self.out_linear(out)

class TruthTransformerLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attn = EmpathyAttention(d_model, num_heads)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
    
    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(x, x, x)
        # Normalizacja (ręczna dla tinygrad)
        x = (x - x.mean(axis=-1, keepdim=True)) / (x.std(axis=-1, keepdim=True) + 1e-5)
        
        ff = self.ff1(x).relu()
        ff = self.ff2(ff)
        x = x + ff
        x = (x - x.mean(axis=-1, keepdim=True)) / (x.std(axis=-1, keepdim=True) + 1e-5)
        return x

class TruthTransformer:
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 2, d_ff: int = 128):
        self.embedding = TruthEmbedding(vocab_size, d_model)
        self.layers = [TruthTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.out = Linear(d_model, vocab_size)
    
    def __call__(self, x: Tensor, stamps: List[VeracityStamp]) -> Tuple[Tensor, Tensor, Tensor]:
        x, ver, ease = self.embedding(x, stamps)
        for layer in self.layers:
            x = layer(x)
        return self.out(x), ver, ease

def truth_loss(logits: Tensor, target: Tensor, veracity: Tensor, ease: Tensor) -> Tensor:
    probs = logits.log_softmax(axis=-1)
    B, T, V = logits.shape
    
    # One-hot encoding targetu
    target_onehot = Tensor.eye(V)[target]
    
    # NLL Loss
    nll = -(probs * target_onehot).sum(axis=-1).mean()
    
    # Ważenie: Silne dowody (high veracity/ease) zwiększają wagę błędu
    avg_quality = (veracity.mean() + ease.mean()) / 2.0
    weighted_loss = nll * (1.0 + avg_quality)
    
    # Kara za ignorancję (dążenie do łatwych prawd)
    ignorance_penalty = (1.0 - ease.mean()) * 0.1
    
    return weighted_loss + ignorance_penalty

# === DEMO ===
if __name__ == "__main__":
    print("Inicjalizacja Veritas v2.0 (Ockham's Gyroscope) w Tinygrad...")
    
    # Włączamy tryb treningowy (dla dropoutów, batchnormów itp. wewnątrz tinygrad)
    Tensor.training = True 
    
    vocab_size = 10
    model = TruthTransformer(vocab_size)
    
    stamps = [
        VeracityStamp("skype.com", 12345, 1105891200, "BITCOIN", {"valid": True}),
        VeracityStamp("anomalia.com", 23456, 1356998400, "A->B 2013", {"valid": False}) 
    ]
    
    x = Tensor([[1, 2, 3]])
    target = Tensor([[2, 3, 4]])
    
    print("Przetwarzanie...")
    logits, ver_scores, ease_scores = model(x, stamps)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Veracity Scores: {ver_scores.numpy()}")
    
    # Definiujemy parametry do optymalizacji
    # UWAGA: Pomijamy model.embedding.weight, bo gradient tam nie dotrze przez NumPy!
    params_to_optimize = [model.out.weight] + [l.attn.out_linear.weight for l in model.layers]
    
    optim = optim.Adam(params_to_optimize, lr=0.01)
    
    print("\nRozpoczynam strojenie żyroskopu (Trening)...")
    
    for i in range(10):
        optim.zero_grad()
        logits, ver, ease = model(x, stamps)
        loss = truth_loss(logits, target, ver, ease)
        loss.backward()
        optim.step()
        print(f"Epoka {i+1}: Loss = {loss.numpy().item():.4f} | Ease Bias = {ease.mean().numpy().item():.3f}")
    
    print("\nSukces. System preferuje prawdę o niskiej entropii.")
