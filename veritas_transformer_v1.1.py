# -*- coding: utf-8 -*-
"""
Truth Transformer v1.1 z Mechanizmem 'Stężenia' Prawdy
Rozszerzenie: Adaptive noise + truth_density metric dla kondensacji emergentnej prawdy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import hashlib
from datetime import datetime

class TruthEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor, stamps: list[str], intention_level: float = 0.5) -> torch.Tensor:
        emb = self.embedding(x)
        for i, stamp in enumerate(stamps):
            parts = dict(p.split(':') for p in stamp.split(';') if ':' in p)
            if 'date' in parts:
                dt = datetime.strptime(parts['date'], '%Y-%m-%d')
                pos = (dt - datetime(2000,1,1)).days
                sin_emb = torch.sin(torch.tensor(pos) / 10000 ** (2 * torch.arange(self.d_model//2) / self.d_model))
                emb[:, i, :self.d_model//2] += sin_emb.to(emb.device)
            if '#' in parts:
                hash_val = int(hashlib.sha256(parts['#'].encode()).hexdigest(), 16) % self.d_model
                emb[:, i] += torch.tensor([hash_val / self.d_model] * self.d_model).to(emb.device)
        
        # Adaptive Noise: Wyższy dla niższej intencji (brute-force kondensacja)
        noise_scale = 0.1 * (1 - intention_level)
        emb += torch.randn_like(emb) * noise_scale
        return emb

class EmpathyAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        Q = self.q_linear(q).view(q.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(k.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(v.size(0), -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        # Empathy Bias: Skalowany cosine sim dla 'stężenia'
        empathy_bias = F.cosine_similarity(Q.mean(dim=2), V.mean(dim=2), dim=-1).unsqueeze(-1).unsqueeze(-1)
        scores += empathy_bias * 0.5
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_probs = self.softmax(scores)
        attn_output = torch.matmul(attn_probs, V).transpose(1, 2).contiguous().view(q.size(0), -1, self.num_heads * self.d_k)
        return self.out_linear(attn_output)

class TruthTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.attention = EmpathyAttention(d_model, num_heads)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class TruthTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 3, d_ff: int = 256):
        super().__init__()
        self.embedding = TruthEmbedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TruthTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.out_linear = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, stamps: list[str], intention_level: float = 0.5) -> torch.Tensor:
        emb = self.embedding(x, stamps, intention_level)
        for layer in self.layers:
            emb = layer(emb)
        return self.out_linear(emb)

# Custom Loss z 'Stężeniem' (Density Metric) - Fixed for token prediction
def truth_loss(output: torch.Tensor, target_truth: torch.Tensor, empathy_scores: torch.Tensor) -> torch.Tensor:
    # NLL loss for token prediction (fixed from MSE)
    nll = 0.0
    log_probs = F.log_softmax(output, dim=-1)
    for i in range(target_truth.shape[1]):
        nll += -log_probs[0, i, target_truth[0, i]]
    nll /= target_truth.shape[1]
    
    reg = torch.mean(1 - empathy_scores)
    
    # Truth Density – średnia cosine sim dla 'stężenia'
    probs = F.softmax(output, dim=-1).squeeze(0)  # (seq, vocab)
    target_onehot = torch.zeros(target_truth.shape[1], output.size(-1))
    for i in range(target_truth.shape[1]):
        target_onehot[i, target_truth[0, i]] = 1.0
    density = F.cosine_similarity(probs, target_onehot, dim=-1).mean()
    density_penalty = 0.2 * (1 - density)  # Penalizuj niskie stężenie
    
    return nll + 0.1 * reg + density_penalty

# Przykład Użycia (Demo z 'Stężeniem')
if __name__ == "__main__":
    vocab = {"URL:": 0, "date:": 1, "#:": 2, "BITCOIN": 3, "genesis": 4}
    vocab_size = len(vocab)
    model = TruthTransformer(vocab_size)
    
    stamps = ["URL:skype.com;date:2005-01-17;#:BITCOIN", "narracja:genesis note 2004", "anomalia:A->B 2013"]
    x = torch.tensor([[0, 1, 2]])
    intention = 0.8  # Wysoka intencja = niski noise
    output = model(x, stamps, intention)
    print(f"Output shape: {output.shape}")
    
    target = torch.tensor([[3, 4, 2]])
    empathy = torch.tensor([0.8, 0.9, 0.7])
    loss = truth_loss(output, target, empathy)
    print(f"Initial Loss with Density: {loss.item():.4f}")
    
    # Trening: Symulacja z poisoned (niski intention = wyższy noise)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(20):
        output = model(x, stamps, intention_level=0.3)  # Niski intention = test adversarza
        loss = truth_loss(output, target, empathy)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        # Compute density for print
        probs = F.softmax(output, dim=-1).squeeze(0)
        target_onehot = torch.zeros(target.shape[1], vocab_size)
        for i in range(target.shape[1]):
            target_onehot[i, target[0, i]] = 1.0
        density = F.cosine_similarity(probs, target_onehot, dim=-1).mean().item()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}, Truth Density {density:.4f}")
