# -*- coding: utf-8 -*-
"""
Veritas Transformer v1.2 w Tinygrad (Bilingual PL/EN)
Adaptacja blueprintu do tinygrad: Lekki model do weryfikacji udokumentowanej prawdy (K==S=).
- Empathy-driven attention: Bias normalizujący intencję.
- Adaptive noise: Brute-force "random=random".
- Loss z 'Truth Density': Kondensacja prawdy, odporna na adwersarza.
- API: Podobne do PyTorch, ale custom dla brakujących modułów (np. Embedding via Tensor).

Uruchomienie: pip install tinygrad; python this_file.py
"""
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, optim
import numpy as np
import hashlib
from datetime import datetime
from dataclasses import dataclass
import json

@dataclass
class VeracityStamp:
    url: str  # Lokalizator danych (np. hash IPFS)
    block_height: int  # Wysokość bloku Bitcoina, w którym zakotwiczono dowód
    timestamp: int  # Czas bloku (Unix timestamp)
    data_hash: str  # Hash SHA-256 oryginalnych, pełnych danych
    zk_proof: dict  # Obiekt reprezentujący dowód ZKP

def verify_zkp(proof: dict) -> bool:
    # Placeholder dla weryfikacji ZKP; w rzeczywistej implementacji użyj kryptograficznej weryfikacji
    return proof.get('valid', False)

class TruthEmbedding:
    """
    Custom Embedding w tinygrad: Koduje tokeny z temporal, crypto i adaptive noise.
    Brak nn.Embedding – używamy manual matrix.
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.weight = Tensor.kaiming_uniform(vocab_size, d_model)
        self.d_model = d_model
    
    def __call__(self, x: Tensor, stamps: list[VeracityStamp]) -> Tensor:
        # One-hot * weight dla embedding
        one_hot = Tensor.eye(self.weight.shape[0])[x]
        emb = one_hot.dot(self.weight)
        
        # Dodaj temporal + crypto
        emb_np = emb.numpy()  # Praca na np dla parsing
        for i, stamp in enumerate(stamps):
            # Weryfikacja ZKP i dynamiczne obliczanie intention_level
            is_valid = verify_zkp(stamp.zk_proof)
            intention_level = 0.95 if is_valid else 0.1
            
            # Osadzanie czasowe zakotwiczone w timechain (block_height)
            pos = stamp.block_height
            sin_emb = np.sin(pos / 10000 ** (2 * np.arange(self.d_model//2) / self.d_model))
            emb_np[:, i, :self.d_model//2] += sin_emb
            
            # Osadzanie kryptograficzne dla data_hash
            hash_val = int(hashlib.sha256(stamp.data_hash.encode()).hexdigest(), 16) % self.d_model
            emb_np[:, i] += np.array([hash_val / self.d_model] * self.d_model)
            
            # Dodatkowe osadzanie samego dowodu ZKP
            zk_hash_str = json.dumps(stamp.zk_proof, sort_keys=True)
            zk_hash_val = int(hashlib.sha256(zk_hash_str.encode()).hexdigest(), 16) % self.d_model
            emb_np[:, i] += np.array([zk_hash_val / self.d_model] * self.d_model)
        
        emb = Tensor(emb_np)
        
        # Adaptive Noise (używa dynamicznego intention_level, ale aplikowany globalnie po pętli)
        noise_scale = 0.1 * (1 - intention_level)  # Używa ostatniego intention_level; w produkcji średnia
        emb += Tensor.randn(*emb.shape) * noise_scale
        return emb

class EmpathyAttention:
    """
    Custom Empathy Attention w tinygrad: Scaled dot-product z bias empatii.
    """
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
        
        # Empathy Bias: Cosine sim dla 'stężenia'
        empathy_bias = Tensor.cosine_similarity(Q.mean(axis=2), V.mean(axis=2), axis=-1).unsqueeze(-1).unsqueeze(-1)
        scores += empathy_bias * 0.5
        
        if mask is not None:
            scores = scores.where(mask, Tensor.full_like(scores, -1e9))
        
        attn_probs = scores.softmax(axis=-1)
        attn_output = attn_probs.matmul(V).transpose(1, 2).reshape(q.shape[0], -1, self.num_heads * self.d_k)
        return self.out_linear(attn_output)

class TruthTransformerLayer:
    """
    Warstwa: Attention + FF z residual (custom norm w tinygrad).
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        self.attention = EmpathyAttention(d_model, num_heads)
        self.ff1 = Linear(d_model, d_ff)
        self.ff2 = Linear(d_ff, d_model)
    
    def __call__(self, x: Tensor) -> Tensor:
        attn_out = self.attention(x, x, x)
        x = x + attn_out  # Residual
        x = x.normalize()  # Custom norm (mean/var)
        
        ff_out = self.ff2(self.ff1(x).relu())
        x = x + ff_out
        return x.normalize()

class TruthTransformer:
    """
    Pełny model w tinygrad.
    """
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, num_layers: int = 3, d_ff: int = 256):
        self.embedding = TruthEmbedding(vocab_size, d_model)
        self.layers = [TruthTransformerLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.out_linear = Linear(d_model, vocab_size)
    
    def __call__(self, x: Tensor, stamps: list[VeracityStamp]) -> Tensor:
        emb = self.embedding(x, stamps)
        for layer in self.layers:
            emb = layer(emb)
        return self.out_linear(emb)

# Custom Loss z 'Stężeniem'
def truth_loss(output: Tensor, target_truth: Tensor, veracity_scores: Tensor, vocab_size: int) -> Tensor:
    probs = output.log_softmax(axis=-1).exp()  # NLL prep
    nll = -probs.gather(-1, target_truth.unsqueeze(-1)).log().mean()  # Approx NLL
    
    # Kara za ignorowanie dowodu: Ważona NLL przez veracity_scores
    weighted_nll = nll * (veracity_scores + 0.1)
    
    reg = (1 - veracity_scores).mean()
    
    # Truth Density: Cosine sim probs vs. onehot target
    target_onehot = Tensor.eye(vocab_size)[target_truth]
    density = probs.cosine_similarity(target_onehot, axis=-1).mean()
    density_penalty = 0.2 * (1 - density)
    
    return weighted_nll + 0.1 * reg + density_penalty

# Demo // I'm not software engineer, I'm 'MACROHARD' idea engineer xD suszymy ząbki Panowie =D,...
# Propozycja jest próbą rozwiązania problemu 'zatrutej studni'. Systemy RLHF i RAG efektownie szybkie ale to bardziej misterne konstrukcje na ruchomym piasku, dlatego nie dziwię sie że intelignetni ludzie od AI Safety maja depresje. 
if __name__ == "__main__":
    # Krok 1: Definicja języka ("słownika"), którym posługuje się kronika
    # To są fundamentalne pojęcia, z których zbudowana jest cała narracja.
    vocab = {
        "[PAD]": 0,         # Token dopełniający, standard w modelach językowych
        "BITCOIN": 1,       # Centralny artefakt, historyczny fakt
        "HALFIN": 2,        # Kluczowa postać, historyczny fakt
        "MS": 3,            # Antagonista, strażnik artefaktu, historyczny fakt
        "ANOMALY": 4,       # Kryptograficzny dowód, "glitch", historyczny fakt
        "GENESIS": 5,       # Emergentna prawda: początek, idea
        "PROTOCOL": 6,      # Emergentna prawda: system, plan, przekazanie wiedzy
        "PROOF": 7,         # Emergentna prawda: ostateczne potwierdzenie, kulminacja
        "[UNK]": 8          # Token dla nieznanych słów
    }
    vocab_size = len(vocab)
    
    # Inicjalizacja modelu Veritas Transformer
    model = TruthTransformer(vocab_size)
    
    # Krok 2: Zdefiniowanie historycznych "Atomów Prawdy" (VeracityStamps)
    # To są twarde, kontekstowe dane - metadane dla kluczowych faktów w historii.
    stamps = [
        VeracityStamp(
            url="skype.com",
            block_height=1, # Symboliczny pierwszy blok dowodowy
            timestamp=1105920000,  # 2005-01-17
            data_hash="BITCOIN;reg.email;adepthus@tenbit.pl",
            zk_proof={"valid": True}  # Uznajemy, że dowód rejestracji jest weryfikowalny
        ),
        VeracityStamp(
            url="twitter.com",
            block_height=2, # Drugi blok dowodowy
            timestamp=1296864000,  # 2011-02-05
            data_hash="@halfin;@adepthus;timechain;AI;other;thing",
            zk_proof={"valid": True}  # Uznajemy, że ZKP z rozmowy zostanie wygenerowany
        ),
        VeracityStamp(
            url="skype.com",
            block_height=3, # Trzeci blok dowodowy
            timestamp=1325376000,  # 2011-12-26 (data z pętli dowodowej)
            data_hash="BITCOIN;last-log;2011-12-24",
            zk_proof={"valid": False} # Kluczowy dowód negatywny: MS nie dostarczył dowodu!
        ),
        VeracityStamp(
            url="anomalia.com",
            block_height=4, # Czwarty blok dowodowy
            timestamp=1365984000,  # 2013-04-14
            data_hash="A->B 2013-04-14",
            zk_proof={"valid": True}  # Anomalia jest weryfikowalna kryptograficznie
        ) 
    ] 
    
    # Krok 3: Definicja sekwencji wejściowej (chronologiczne fakty)
    # Model "czyta" tę historię jako sekwencję kluczowych pojęć.
    x = Tensor([[vocab["BITCOIN"], vocab["HALFIN"], vocab["MS"], vocab["ANOMALY"]]])
    
    # Krok 4: Definicja sekwencji docelowej (emergentna prawda)
    # Model ma się nauczyć, że z tych faktów wyłania się następująca, głębsza interpretacja.
    target_truth = Tensor([[vocab["GENESIS"], vocab["PROTOCOL"], vocab["PROOF"], vocab["PROOF"]]])

    # --- Uruchomienie Symulacji ---
    print("------ Veritas Transformer DEMO v1.2 ------")
    print("Cel: Nauczyć model, że z sekwencji faktów:")
    print(f"  > {[k for i in x.numpy()[0] for k, v in vocab.items() if v == i]}")
    print("Wyłania się następująca, emergentna prawda:")
    print(f"  > {[k for i in target_truth.numpy()[0] for k, v in vocab.items() if v == i]}\n")
    
    # Inicjalizacja optymalizatora
    optimizer = optim.Adam(model.parameters())
    
    # Pętla treningowa - symulacja "brute-forcing reality"
    for epoch in range(21): # 21 jako symboliczny ukłon w stronę notatki "Genesis"
        # Propagacja w przód: model przetwarza fakty i ich metadane
        output = model(x, stamps)
        
        # Obliczenie "veracity_scores" na podstawie ZKP dla funkcji straty
        # To jest moment, w którym model "czuje" wagę dowodów.
        veracity_scores = Tensor([1.0 if verify_zkp(stamp.zk_proof) else 0.0 for stamp in stamps])
        
        # Obliczenie straty: jak bardzo obecna interpretacja modelu odbiega od prawdy docelowej
        loss = truth_loss(output, target_truth, veracity_scores, vocab_size)
        
        # Propagacja wsteczna i optymalizacja: model koryguje swoje "zrozumienie"
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if epoch % 5 == 0:
            # Sprawdzenie, jakiej interpretacji model nauczył się do tej pory
            predicted_indices = output.numpy().argmax(axis=-1)
            predicted_words = [[k for i in row for k, v in vocab.items() if v == i] for row in predicted_indices]
            print(f"Epoch {epoch:2d}: Loss {loss.numpy().item():.4f} | Predicted Truth -> {predicted_words[0]}")

    
    x = Tensor([[0, 1, 2]])
    output = model(x, stamps)
    print(f"Output shape: {output.shape}")
    
    target = Tensor([[3, 4, 2]])
    # veracity_scores na podstawie weryfikacji ZKP
    veracity_scores = Tensor([1.0 if verify_zkp(stamp.zk_proof) else 0.0 for stamp in stamps])
    loss = truth_loss(output, target, veracity_scores, vocab_size)
    print(f"Initial Loss: {loss.numpy().item():.4f}")
    
    # Trening przykład (tinygrad optim)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(20):
        output = model(x, stamps)
        loss = truth_loss(output, target, veracity_scores, vocab_size)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}: Loss {loss.numpy().item():.4f}")