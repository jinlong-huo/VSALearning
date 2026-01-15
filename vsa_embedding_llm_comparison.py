"""
VSA Embedding Comparison for Language Modeling

Tests all torchhd.embeddings methods on a text corpus (Shakespeare)
to determine which VSA embedding is best for LLM applications.

Embeddings tested:
- Standard: nn.Embedding (baseline)
- Projection: Random projection matrix
- Sinusoid: Nonlinear random projection with sin/cos
- Level: Quantization-based encoding    # Config
    d_model = 64  # Reduced for faster training
    n_layers = 1  # Reduced for faster training
    n_heads = 2  # Reduced for faster training
    seq_len = 32  # Reduced for faster training
    batch_size = 64  # Increased batch size
    n_epochs = 10  # Reduced epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')ometer: Thermometer encoding
- Density: intRVFL density encoding
- Random: Random hypervector lookup

Evaluation:
- Cross-entropy loss (perplexity)
- Text generation quality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchhd
from torchhd import embeddings
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math


def to_tensor(x):
    """Convert MAPTensor/VSATensor to regular torch.Tensor"""
    if hasattr(x, 'as_subclass'):
        return x.as_subclass(torch.Tensor)
    return x


# ============================================================================
# EMBEDDING IMPLEMENTATIONS FOR LLM
# ============================================================================

class StandardEmbedding(nn.Module):
    """Standard nn.Embedding (baseline)"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)


class TorchHD_RandomEmbed(nn.Module):
    """Using torchhd.embeddings.Random for token lookup"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='MAP'
        )
    
    def forward(self, x):
        out = self.embed(x)
        return to_tensor(out)


class TorchHD_LevelEmbed(nn.Module):
    """Using torchhd.embeddings.Level - treats token IDs as continuous values"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = embeddings.Level(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='MAP',
            low=0.0,
            high=float(vocab_size - 1)
        )
    
    def forward(self, x):
        # Convert token IDs to float for Level encoding
        x_float = x.float()
        out = self.embed(x_float)
        return to_tensor(out)


class TorchHD_ThermometerEmbed(nn.Module):
    """Using torchhd.embeddings.Thermometer"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Thermometer(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='MAP',
            low=0.0,
            high=float(vocab_size - 1)
        )
    
    def forward(self, x):
        x_float = x.float()
        out = self.embed(x_float)
        return to_tensor(out)


class TorchHD_CircularEmbed(nn.Module):
    """Using torchhd.embeddings.Circular for cyclic token relationships"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Circular(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='MAP'
        )
    
    def forward(self, x):
        x_float = x.float()
        out = self.embed(x_float)
        return to_tensor(out)


class TorchHD_ProjectionEmbed(nn.Module):
    """
    Using torchhd.embeddings.Projection
    First embed with standard embedding, then project to HD space
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.base_embed = nn.Embedding(vocab_size, d_model)
        self.projection = embeddings.Projection(
            in_features=d_model,
            out_features=d_model,
            vsa='MAP'
        )
    
    def forward(self, x):
        base = self.base_embed(x)  # [batch, seq, d_model]
        out = self.projection(base)
        return to_tensor(out)


class TorchHD_SinusoidEmbed(nn.Module):
    """
    Using torchhd.embeddings.Sinusoid
    First embed with standard embedding, then apply sinusoid projection
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.base_embed = nn.Embedding(vocab_size, d_model)
        self.sinusoid = embeddings.Sinusoid(
            in_features=d_model,
            out_features=d_model,
            vsa='MAP'
        )
    
    def forward(self, x):
        base = self.base_embed(x)  # [batch, seq, d_model]
        out = self.sinusoid(base)
        return to_tensor(out)


class TorchHD_DensityEmbed(nn.Module):
    """
    Using torchhd.embeddings.Density (intRVFL)
    First embed with standard embedding, then apply density encoding
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.base_embed = nn.Embedding(vocab_size, d_model)
        self.density = embeddings.Density(
            in_features=d_model,
            out_features=d_model,
            vsa='MAP',
            low=-2.0,
            high=2.0
        )
    
    def forward(self, x):
        base = self.base_embed(x)
        out = self.density(base)
        return to_tensor(out)


# ============================================================================
# SIMPLE LANGUAGE MODEL
# ============================================================================

class SimpleLM(nn.Module):
    """Simple transformer-like LM for testing embeddings"""
    def __init__(self, embedding_layer, vocab_size, d_model, n_layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding (varies by method)
        self.token_embedding = embedding_layer
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Causal mask
        self.register_buffer('causal_mask', None)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        # Get embeddings
        tok_emb = self.token_embedding(x)  # [batch, seq, d_model]
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb
        
        # Create causal mask
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.causal_mask = mask
        
        # Transformer
        x = self.transformer(x, mask=self.causal_mask)
        
        # Output projection
        logits = self.output_proj(x)
        
        # Compute loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )
        
        return logits, loss


# ============================================================================
# DATASET
# ============================================================================

class CharDataset(Dataset):
    """Character-level dataset"""
    def __init__(self, text, char_to_idx, seq_len=64):
        self.seq_len = seq_len
        self.char_to_idx = char_to_idx
        self.data = [char_to_idx.get(ch, 0) for ch in text]
    
    def __len__(self):
        return len(self.data) - self.seq_len - 1
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return x, y


def build_vocab(text):
    """Build character vocabulary"""
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, targets=y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_text(model, start_text, char_to_idx, idx_to_char, device, max_len=100, temperature=0.8):
    """Generate text from model"""
    model.eval()
    
    # Encode start text
    input_ids = [char_to_idx.get(ch, 0) for ch in start_text]
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = list(start_text)
    
    with torch.no_grad():
        for _ in range(max_len):
            # Get predictions
            logits, _ = model(input_ids[:, -64:])  # Use last 64 tokens
            next_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
            next_char = idx_to_char.get(next_token.item(), '')
            generated.append(next_char)
    
    return ''.join(generated)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("VSA EMBEDDING COMPARISON FOR LANGUAGE MODELING")
    print("=" * 80)
    
    # Shakespeare text
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life.
    """ * 10  # Repeat for more training data
    
    # Config
    d_model = 64
    n_layers = 1
    n_heads = 2
    seq_len = 32
    batch_size = 64
    n_epochs = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Config: d_model={d_model}, n_layers={n_layers}, seq_len={seq_len}")
    
    # Build vocabulary
    print("\n[1/4] Building vocabulary...")
    char_to_idx, idx_to_char = build_vocab(sample_text)
    vocab_size = len(char_to_idx)
    print(f"✓ Vocab size: {vocab_size}")
    print(f"✓ Characters: {list(char_to_idx.keys())[:20]}...")
    
    # Create dataset
    print("\n[2/4] Creating dataset...")
    dataset = CharDataset(sample_text, char_to_idx, seq_len=seq_len)
    
    # Split train/test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(f"✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Embedding methods
    print("\n[3/4] Initializing models...")
    embedding_configs = {
        'Standard': lambda: StandardEmbedding(vocab_size, d_model),
        'Random': lambda: TorchHD_RandomEmbed(vocab_size, d_model),
        'Level': lambda: TorchHD_LevelEmbed(vocab_size, d_model),
        'Thermometer': lambda: TorchHD_ThermometerEmbed(vocab_size, d_model),
        'Circular': lambda: TorchHD_CircularEmbed(vocab_size, d_model),
        'Projection': lambda: TorchHD_ProjectionEmbed(vocab_size, d_model),
        'Sinusoid': lambda: TorchHD_SinusoidEmbed(vocab_size, d_model),
        'Density': lambda: TorchHD_DensityEmbed(vocab_size, d_model),
    }
    
    models = {}
    optimizers = {}
    training_history = {}
    
    for name, embed_fn in embedding_configs.items():
        try:
            embed_layer = embed_fn()
            model = SimpleLM(embed_layer, vocab_size, d_model, n_layers, n_heads).to(device)
            models[name] = model
            optimizers[name] = torch.optim.AdamW(model.parameters(), lr=3e-4)
            training_history[name] = {'train': [], 'test': [], 'ppl': []}
            n_params = sum(p.numel() for p in model.parameters())
            print(f"✓ {name:<12}: {n_params:>8,} params")
        except Exception as e:
            print(f"✗ {name:<12}: Failed - {e}")
    
    # Training
    print("\n[4/4] Training...")
    print("-" * 80)
    print(f"{'Epoch':<8}", end="")
    for name in models.keys():
        print(f"{name:<12}", end="")
    print()
    print("-" * 80)
    
    for epoch in range(n_epochs):
        epoch_results = {}
        
        for name, model in models.items():
            train_loss = train_epoch(model, train_loader, optimizers[name], device)
            test_loss, ppl = evaluate(model, test_loader, device)
            
            training_history[name]['train'].append(train_loss)
            training_history[name]['test'].append(test_loss)
            training_history[name]['ppl'].append(ppl)
            epoch_results[name] = test_loss
        
        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"{epoch+1:<8}", end="")
            for name in models.keys():
                print(f"{epoch_results[name]:<12.4f}", end="")
            print()
    
    # Final Results
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    
    final_results = {}
    for name, model in models.items():
        test_loss, ppl = evaluate(model, test_loader, device)
        final_results[name] = {'loss': test_loss, 'perplexity': ppl}
    
    print(f"\n{'Method':<12} {'Loss':<12} {'Perplexity':<12}")
    print("-" * 40)
    for name in models.keys():
        r = final_results[name]
        print(f"{name:<12} {r['loss']:<12.4f} {r['perplexity']:<12.2f}")
    
    # Comparison with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH STANDARD BASELINE")
    print("=" * 80)
    
    baseline_loss = final_results['Standard']['loss']
    print(f"\n{'Method':<12} {'Loss Ratio':<12} {'Status':<20}")
    print("-" * 50)
    
    for name in models.keys():
        ratio = final_results[name]['loss'] / baseline_loss
        if ratio <= 1.0:
            status = "BETTER"
        elif ratio <= 1.05:
            status = "COMPARABLE"
        elif ratio <= 1.2:
            status = "ACCEPTABLE"
        else:
            status = "WORSE"
        print(f"{name:<12} {ratio:<12.2f}x {status}")
    
    # Best method
    best = min(final_results.keys(), key=lambda k: final_results[k]['loss'])
    print(f"\n>>> Best embedding: {best} (Loss: {final_results[best]['loss']:.4f}, PPL: {final_results[best]['perplexity']:.2f})")
    
    # Text generation comparison
    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)
    
    prompt = "To be, or not"
    print(f"\nPrompt: '{prompt}'")
    print("-" * 60)
    
    for name in ['Standard', 'Sinusoid', 'Projection', best]:
        if name in models:
            generated = generate_text(models[name], prompt, char_to_idx, idx_to_char, device, max_len=80)
            print(f"\n{name}:")
            print(f"  {generated[:100]}...")
    
    # Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    visualize_results(training_history, final_results, models.keys())
    
    return final_results, training_history


def visualize_results(history, results, method_names):
    """Generate comparison visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    for name in method_names:
        ax.plot(history[name]['test'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Training Progress (Test Loss)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Final loss comparison
    ax = axes[0, 1]
    names = list(method_names)
    losses = [results[n]['loss'] for n in names]
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    bars = ax.bar(names, losses, color=colors, edgecolor='black')
    ax.set_ylabel('Test Loss (lower is better)')
    ax.set_title('Final Test Loss Comparison')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Perplexity comparison
    ax = axes[1, 0]
    ppls = [results[n]['perplexity'] for n in names]
    bars = ax.bar(names, ppls, color=colors, edgecolor='black')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title('Final Perplexity Comparison')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: Relative performance
    ax = axes[1, 1]
    baseline = results['Standard']['loss']
    ratios = [results[n]['loss'] / baseline for n in names]
    colors_ratio = ['green' if r <= 1.0 else 'orange' if r <= 1.1 else 'red' for r in ratios]
    bars = ax.bar(names, ratios, color=colors_ratio, edgecolor='black', alpha=0.7)
    ax.axhline(y=1.0, color='black', linestyle='--', label='Baseline')
    ax.set_ylabel('Loss Ratio vs Standard')
    ax.set_title('Relative Performance (1.0 = Standard)')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('vsa_embedding_llm_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: vsa_embedding_llm_comparison.png")
    
    plt.close()


if __name__ == "__main__":
    results, history = main()
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR VSA-LLM")
    print("=" * 80)
    print("""
Based on the comparison results:

1. If Sinusoid/Projection perform well:
   -> Use them as drop-in replacement in vsa_llm_train.py
   -> Benefits: Better feature separation, random Fourier features

2. If Standard performs best:
   -> Keep nn.Embedding, focus VSA on other components
   -> Use VSA for: positional encoding, knowledge memory

3. For production VSA-LLM:
   -> Combine best embedding with VSA positional binding
   -> Add VSA knowledge memory for one-shot learning
""")
