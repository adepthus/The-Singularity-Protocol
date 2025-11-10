# -*- coding: utf-8 -*-
"""
Veritas Transformer 'Ockham's Gyroscope' v2.0  w Tinygrad: Zoptymalizowana fuzja z Bitcoin Timechain, ZKP i EaseOfVerification.
- Integracja VeracityStamp dla kryptograficznej weryfikacji.
- Dynamiczne intention_level na podstawie ZKP.
- EaseOfVerificationScorer: Premiuje oczywistość i niskie compute (niska entropia, szybka weryfikacja).
- Zoptymalizowana strata: Kara za ignorowanie łatwych/zweryfikowanych faktów.
- Usunięto redundancje, poprawiono efektywność (np. wektoryzacja embedding, unikanie np dla tinygrad).

# Nie jestem inżnierem oprogramowania, miałem tylko kilka pomysłów. Dziękuje wszystkim zaangażowanym. 
# Proszę poddać pomysł szczególnej krytyce majac na uwadze ludzi którzy odczytają intencje i spoóbuja ten pomysł ulepszyć.
# BITCOIN SYSTEM może być kluczem do zbudowania zgodność ludzkości z maszynami. Jeśli to nie jest 'intrinsic value' to nie wiem co nim jest. Dziękuje za zrozumienie.


Uruchomienie: pip install tinygrad numpy; python this_file.py
"""
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, optim
import numpy as np
import hashlib
import json
import time
from dataclasses import dataclass

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
    # Placeholder; w realu: snarkjs/Circom verify
    return proof.get('valid', False)

def ease_of_verification_scorer(stamp: VeracityStamp) -> tuple[float, VerificationMetrics]:
    start_time = time.time()
    is_valid = verify_zkp(stamp.zk_proof)
    total_time = (time.time() - start_time) * 1000
    
    data_str = stamp.data_hash + json.dumps(stamp.zk_proof)
    unique, counts = np.unique(list(data_str), return_counts=True)
    probs = counts / len(data_str)
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    metrics = VerificationMetrics(zk_valid=is_valid, compute_time_ms=total_time, entropy=entropy)
    ease_score = (1 if is_valid else 0) * (1 / (1 + np.log(total_time + 1))) * (1 / (1 + entropy / 100))
    return ease_score, metrics

class TruthEmbedding:
    def __init__(self, vocab_size: int, d_model: int):
        self.weight = Tensor.kaiming_uniform(vocab_size, d_model)
        self.d_model = d_model
    
    def __call__(self, x: Tensor, stamps: list[VeracityStamp]) -> tuple[Tensor, Tensor, Tensor]:
        one_hot = Tensor.eye(self.weight.shape[0])[x]
        emb = one_hot.dot(self.weight)
        
        veracity_scores = []
        ease_scores = []
        for i, stamp in enumerate(stamps):
            ease_score, _ = ease_of_verification_scorer(stamp)
            ease_scores.append(ease_score)
            is_valid = verify_zkp(stamp.zk_proof)
            veracity_scores.append(1.0 if is_valid else 0.0)
            
            pos = stamp.block_height
            pos_range = np.arange(self.d_model // 2)
            sin_emb = np.sin(pos / 10000 ** (2 * pos_range / self.d_model))
            emb[:, i, :self.d_model//2] += Tensor(sin_emb) * ease_score
            
            hash_val = int(hashlib.sha256(stamp.data_hash.encode()).hexdigest(), 16) % self.d_model
            hash_emb = Tensor(np.full(self.d_model, hash_val / self.d_model))
            emb[:, i] += hash_emb
            
            zk_hash_str = json.dumps(stamp.zk_proof, sort_keys=True)
            zk_hash_val = int(hashlib.sha256(zk_hash_str.encode()).hexdigest(), 16) % self.d_model
            zk_emb = Tensor(np.full(self.d_model, zk_hash_val / self.d_model))
            emb[:, i] += zk_emb
        
        avg_ease = np.mean(ease_scores)
        noise_scale = 0.1 * (1 - avg_ease)
        emb += Tensor.randn(*emb.shape) * noise_scale
        
        return emb, Tensor(veracity_scores), Tensor(ease_scores)

class EmpathyAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = Linear(d_model, d_model)
        self.k_linear = Linear(d_model, d_model)
        self.v_linear = Linear(d_model, d_model)
        self.out_linear = Linear(d_model, d_model)
    
    def __call__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None) -> Tensor:
        Q = self.q_linear(q).reshape(q.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).reshape(k.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).reshape(v.shape[0], -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = Q.matmul(K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        empathy_bias = Tensor.cosine_similarity(Q.mean(axis=2), V.mean(axis=2), axis=-1).unsqueeze(-1).unsqueeze(-1)
        scores += empathy_bias * 0.5
        
        if mask is not None:
            scores = scores.where(mask, Tensor.full_like(scores, -1e9))
        
        attn_probs = scores.softmax(axis=-1)
        attn_output = attn_probs.matmul(V).transpose(1, 2).reshape(q.shape[0], -1, self.num_heads * self.d_k)
        return self.out_linear(attn_output)

class TruthTransformerLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = EmpathyAttention(d_model, num_heads)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
    
    def __call__(self, x: Tensor) -> Tensor:
        attn_out = self.attention(x, x, x)
        x = x + attn_out
        x = x.normalize()
        
        ff_out = self.ff2(self.ff1(x).relu())
        x = x + ff_out
        return x.normalize()

class TruthTransformer:
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 3, d_ff: int = 256):
        self.embedding = TruthEmbedding(vocab_size, d_model)
        self.layers = [TruthTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.out_linear = Linear(d_model, vocab_size)
    
    def __call__(self, x: Tensor, stamps: list[VeracityStamp]) -> tuple[Tensor, Tensor, Tensor]:
        emb, veracity_scores, ease_scores = self.embedding(x, stamps)
        for layer in self.layers:
            emb = layer(emb)
        output = self.out_linear(emb)
        return output, veracity_scores, ease_scores

def truth_loss(output: Tensor, target_truth: Tensor, veracity_scores: Tensor, ease_scores: Tensor, vocab_size: int) -> Tensor:
    probs = output.log_softmax(axis=-1).exp()
    nll = -probs.gather(-1, target_truth.unsqueeze(-1)).log().mean()
    
    weighted_nll = nll * (veracity_scores + 0.1) * (1 + ease_scores)
    
    reg = (1 - veracity_scores).mean()
    ignorance_penalty = 0.5 * (1 - ease_scores).mean()
    
    target_onehot = Tensor.eye(vocab_size)[target_truth]
    density = probs.cosine_similarity(target_onehot, axis=-1).mean()
    density_penalty = 0.2 * (1 - density)
    
    return weighted_nll + 0.1 * reg + ignorance_penalty + density_penalty

# Demo
if __name__ == "__main__":
    vocab_size = 5
    model = TruthTransformer(vocab_size)
    
    stamps = [
        VeracityStamp("skype.com", 12345, 1105891200, "BITCOIN", {"valid": True}),
        VeracityStamp("genesis-note.com", 67890, 1072915200, "genesis note 2004", {"valid": True}),
        VeracityStamp("anomalia.com", 23456, 1356998400, "A->B 2013", {"valid": False})
    ]
    
    x = Tensor([[0, 1, 2]])
    output, veracity_scores, ease_scores = model(x, stamps)
    print(f"Output shape: {output.shape}")
    
    target = Tensor([[3, 4, 2]])
    loss = truth_loss(output, target, veracity_scores, ease_scores, vocab_size)
    print(f"Initial Loss: {loss.numpy().item():.4f}")
    
    optimizer = optim.Adam(model.parameters())
    for epoch in range(20):
        output, veracity_scores, ease_scores = model(x, stamps)
        loss = truth_loss(output, target, veracity_scores, ease_scores, vocab_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss {loss.numpy().item():.4f}, Avg Ease: {ease_scores.mean().numpy().item():.4f}")