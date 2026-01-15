"""
VSA Tensor Type Comparison for Language Modeling

This script compares different VSA TENSOR TYPES (not embedding methods) 
to determine which VSA architecture works best for LLM applications.

VSA Tensor Types (from torchhd.types.VSAOptions):
- MAP: Multiply Add Permute (dense bipolar, elements from {-1, 1})
- HRR: Holographic Reduced Representations (real-valued, FFT-based binding)
- FHRR: Fourier HRR (complex-valued, phase-based)
- BSC: Binary Spatter Codes (binary {0, 1})
- BSBC: Block Sparse Binary Codes
- VTB: Vector-Derived Transformation Binding
- MCR: Modular Composite Representations  
- CGR: Cyclic Group Representations

Key Differences:
- MAP: Element-wise multiplication for binding, addition for bundling
- HRR: Circular convolution for binding (via FFT), addition for bundling
- FHRR: Element-wise complex multiplication, addition for bundling
- BSC: XOR for binding, majority vote for bundling

We use embeddings.Random with different vsa= parameters to test each type.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchhd import embeddings
import numpy as np
import matplotlib.pyplot as plt
import math
import warnings
warnings.filterwarnings('ignore')


def to_tensor(x):
    """Convert VSATensor to regular torch.Tensor for compatibility with nn layers"""
    if hasattr(x, 'as_subclass'):
        return x.as_subclass(torch.Tensor)
    return x


def to_real(x):
    """Convert complex tensor to real (for FHRR compatibility)"""
    if x.is_complex():
        # Stack real and imaginary parts, or take real part
        return torch.real(x)
    return x


# ============================================================================
# VSA TENSOR TYPE EMBEDDINGS
# ============================================================================

class StandardEmbedding(nn.Module):
    """Standard learnable nn.Embedding (baseline)"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embed(x)


class VSA_MAP_Embedding(nn.Module):
    """
    MAP: Multiply Add Permute
    - Dense bipolar vectors {-1, 1}
    - Binding: element-wise multiplication
    - Bundling: element-wise addition
    - Good for: general purpose, robust to noise
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='MAP',
            requires_grad=False
        )
        # Learnable scaling to help gradients
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x):
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_HRR_Embedding(nn.Module):
    """
    HRR: Holographic Reduced Representations  
    - Real-valued continuous vectors
    - Binding: circular convolution (via FFT)
    - Bundling: element-wise addition
    - Good for: analogical reasoning, smooth similarity
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='HRR',
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_FHRR_Embedding(nn.Module):
    """
    FHRR: Fourier Holographic Reduced Representations
    - Complex-valued vectors (phase-based)
    - Binding: element-wise complex multiplication
    - Bundling: element-wise addition
    - Good for: resonator networks, iterative retrieval
    
    Note: Outputs complex values, we take real part for transformer compatibility
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # FHRR uses complex numbers, so d_model should represent real dimensions
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='FHRR',
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        # Project complex to real
        self.proj = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        out = self.embed(x)
        out = to_tensor(out)
        # Convert complex to real by stacking real and imaginary parts
        if out.is_complex():
            real_part = torch.real(out)
            imag_part = torch.imag(out)
            out = torch.cat([real_part, imag_part], dim=-1)
            out = self.proj(out)
        return out * self.scale


class VSA_BSC_Embedding(nn.Module):
    """
    BSC: Binary Spatter Codes
    - Binary vectors {0, 1} (or {True, False})
    - Binding: XOR operation
    - Bundling: majority vote (thresholded sum)
    - Good for: memory efficiency, fast operations
    
    Note: Binary values need to be converted to float for transformers
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='BSC',
            requires_grad=False
        )
        # Convert binary {0,1} to {-1,1} range for better gradient flow
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        out = self.embed(x)
        out = to_tensor(out).float()
        # Convert {0,1} to {-1,1}
        out = out * 2.0 - 1.0
        return out * self.scale


class VSA_VTB_Embedding(nn.Module):
    """
    VTB: Vector-derived Transformation Binding
    - Real-valued vectors
    - Binding: matrix transformation derived from vectors
    - Good for: structured knowledge representation
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = embeddings.Random(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            vsa='VTB',
            requires_grad=False
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_Sinusoid_MAP_Embedding(nn.Module):
    """
    Sinusoid embedding with MAP VSA type
    - Nonlinear random projection with sin/cos activation
    - Uses MAP hypervectors as base
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # Use a base embedding then sinusoid projection
        self.base_embed = nn.Embedding(vocab_size, d_model)
        self.sinusoid = embeddings.Sinusoid(
            in_features=d_model,
            out_features=d_model,
            vsa='MAP'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.3)
    
    def forward(self, x):
        base = self.base_embed(x)
        out = self.sinusoid(base)
        return to_tensor(out) * self.scale


class VSA_Sinusoid_HRR_Embedding(nn.Module):
    """
    Sinusoid embedding with HRR VSA type
    - Combines sinusoid projection with HRR hypervectors
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.base_embed = nn.Embedding(vocab_size, d_model)
        self.sinusoid = embeddings.Sinusoid(
            in_features=d_model,
            out_features=d_model,
            vsa='HRR'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        base = self.base_embed(x)
        out = self.sinusoid(base)
        return to_tensor(out) * self.scale


# ============================================================================
# SIMPLE LANGUAGE MODEL
# ============================================================================

class SimpleLM(nn.Module):
    """Simple transformer LM for testing different VSA embeddings"""
    def __init__(self, embedding_layer, vocab_size, d_model, n_layers=1, n_heads=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embedding (the VSA type we're testing)
        self.token_embedding = embedding_layer
        
        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(512, d_model)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        self.register_buffer('causal_mask', None)
    
    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        
        # Embeddings
        tok_emb = self.token_embedding(x)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(positions)
        x = tok_emb + pos_emb
        
        # Causal mask
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            self.causal_mask = mask
        
        # Transformer
        x = self.transformer(x, mask=self.causal_mask)
        
        # Output
        logits = self.output_proj(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        
        return logits, loss


# ============================================================================
# DATASET
# ============================================================================

class CharDataset(Dataset):
    def __init__(self, text, char_to_idx, seq_len=32):
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
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


# ============================================================================
# TRAINING
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
    perplexity = math.exp(min(avg_loss, 20))  # Clip to prevent overflow
    return avg_loss, perplexity


def generate_text(model, start_text, char_to_idx, idx_to_char, device, max_len=80, temperature=0.8):
    model.eval()
    input_ids = [char_to_idx.get(ch, 0) for ch in start_text]
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated = list(start_text)
    
    with torch.no_grad():
        for _ in range(max_len):
            logits, _ = model(input_ids[:, -32:])
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            next_char = idx_to_char.get(next_token.item(), '')
            generated.append(next_char)
    
    return ''.join(generated)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(history, results, method_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training curves
    ax = axes[0, 0]
    for name in method_names:
        if name in history:
            ax.plot(history[name]['test'], label=name, linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Training Progress by VSA Type')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Final loss comparison
    ax = axes[0, 1]
    names = [n for n in method_names if n in results]
    losses = [results[n]['loss'] for n in names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, losses, color=colors, edgecolor='black')
    ax.set_ylabel('Test Loss (lower is better)')
    ax.set_title('Final Test Loss by VSA Type')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Perplexity comparison
    ax = axes[1, 0]
    ppls = [results[n]['perplexity'] for n in names]
    bars = ax.bar(names, ppls, color=colors, edgecolor='black')
    ax.set_ylabel('Perplexity (lower is better)')
    ax.set_title('Final Perplexity by VSA Type')
    ax.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Relative performance
    ax = axes[1, 1]
    if 'Standard' in results:
        baseline = results['Standard']['loss']
        ratios = [results[n]['loss'] / baseline for n in names]
        colors_ratio = ['green' if r <= 1.0 else 'orange' if r <= 1.2 else 'red' for r in ratios]
        bars = ax.bar(names, ratios, color=colors_ratio, edgecolor='black', alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', label='Standard Baseline')
        ax.set_ylabel('Loss Ratio vs Standard')
        ax.set_title('Relative Performance (1.0 = Standard)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig('output/vsa_tensor_type_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: output/vsa_tensor_type_comparison.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("VSA TENSOR TYPE COMPARISON FOR LANGUAGE MODELING")
    print("=" * 80)
    print("""
    Testing different VSA ARCHITECTURES (not just embedding methods):
    - Standard: Regular nn.Embedding (baseline)
    - MAP: Multiply Add Permute (bipolar {-1,1})
    - HRR: Holographic Reduced Representations (real-valued, FFT binding)
    - FHRR: Fourier HRR (complex-valued, phase encoding)
    - BSC: Binary Spatter Codes (binary {0,1}, XOR binding)
    - VTB: Vector-derived Transformation Binding
    - Sinusoid+MAP: Sinusoid projection with MAP
    - Sinusoid+HRR: Sinusoid projection with HRR
    """)
    
    # Shakespeare text
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die, to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to. 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep: perchance to dream: ay, there's the rub;
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause: there's the respect
    That makes calamity of so long life.
    
    Friends, Romans, countrymen, lend me your ears;
    I come to bury Caesar, not to praise him.
    The evil that men do lives after them;
    The good is oft interred with their bones.
    """ * 5
    
    # Config
    d_model = 64
    n_layers = 1
    n_heads = 2
    seq_len = 32
    batch_size = 64
    n_epochs = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Config: d_model={d_model}, n_layers={n_layers}, n_epochs={n_epochs}")
    
    # Build vocabulary
    print("\n[1/4] Building vocabulary...")
    char_to_idx, idx_to_char = build_vocab(sample_text)
    vocab_size = len(char_to_idx)
    print(f"✓ Vocab size: {vocab_size}")
    
    # Create dataset
    print("\n[2/4] Creating dataset...")
    dataset = CharDataset(sample_text, char_to_idx, seq_len=seq_len)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print(f"✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # VSA Tensor Type configurations
    print("\n[3/4] Initializing VSA types...")
    vsa_configs = {
        'Standard': lambda: StandardEmbedding(vocab_size, d_model),
        'MAP': lambda: VSA_MAP_Embedding(vocab_size, d_model),
        'HRR': lambda: VSA_HRR_Embedding(vocab_size, d_model),
        'FHRR': lambda: VSA_FHRR_Embedding(vocab_size, d_model),
        'BSC': lambda: VSA_BSC_Embedding(vocab_size, d_model),
        'VTB': lambda: VSA_VTB_Embedding(vocab_size, d_model),
        'Sinusoid+MAP': lambda: VSA_Sinusoid_MAP_Embedding(vocab_size, d_model),
        'Sinusoid+HRR': lambda: VSA_Sinusoid_HRR_Embedding(vocab_size, d_model),
    }
    
    models = {}
    optimizers = {}
    training_history = {}
    
    for name, config_fn in vsa_configs.items():
        try:
            embed_layer = config_fn()
            model = SimpleLM(embed_layer, vocab_size, d_model, n_layers, n_heads).to(device)
            models[name] = model
            optimizers[name] = torch.optim.AdamW(model.parameters(), lr=3e-4)
            training_history[name] = {'train': [], 'test': [], 'ppl': []}
            n_params = sum(p.numel() for p in model.parameters())
            print(f"✓ {name:<15}: {n_params:>8,} params")
        except Exception as e:
            print(f"✗ {name:<15}: Failed - {e}")
    
    # Training
    print("\n[4/4] Training...")
    print("-" * 90)
    header = f"{'Epoch':<8}"
    for name in models.keys():
        header += f"{name:<12}"
    print(header)
    print("-" * 90)
    
    for epoch in range(n_epochs):
        epoch_results = {}
        
        for name, model in models.items():
            try:
                train_loss = train_epoch(model, train_loader, optimizers[name], device)
                test_loss, ppl = evaluate(model, test_loader, device)
                
                training_history[name]['train'].append(train_loss)
                training_history[name]['test'].append(test_loss)
                training_history[name]['ppl'].append(ppl)
                epoch_results[name] = test_loss
            except Exception as e:
                epoch_results[name] = float('inf')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            row = f"{epoch+1:<8}"
            for name in models.keys():
                row += f"{epoch_results.get(name, float('inf')):<12.4f}"
            print(row)
    
    # Final Results
    print("\n" + "=" * 80)
    print("FINAL RESULTS: VSA TENSOR TYPE COMPARISON")
    print("=" * 80)
    
    final_results = {}
    for name, model in models.items():
        try:
            test_loss, ppl = evaluate(model, test_loader, device)
            final_results[name] = {'loss': test_loss, 'perplexity': ppl}
        except:
            pass
    
    print(f"\n{'VSA Type':<15} {'Loss':<12} {'Perplexity':<12} {'vs Standard':<12}")
    print("-" * 55)
    
    baseline_loss = final_results.get('Standard', {}).get('loss', 1.0)
    for name in models.keys():
        if name in final_results:
            r = final_results[name]
            ratio = r['loss'] / baseline_loss
            print(f"{name:<15} {r['loss']:<12.4f} {r['perplexity']:<12.2f} {ratio:<12.2f}x")
    
    # Best VSA type
    vsa_only = {k: v for k, v in final_results.items() if k != 'Standard'}
    if vsa_only:
        best_vsa = min(vsa_only.keys(), key=lambda k: vsa_only[k]['loss'])
        print(f"\n🏆 BEST VSA TYPE: {best_vsa}")
        print(f"   Loss: {vsa_only[best_vsa]['loss']:.4f}")
        print(f"   Perplexity: {vsa_only[best_vsa]['perplexity']:.2f}")
        print(f"   vs Standard: {vsa_only[best_vsa]['loss']/baseline_loss:.2f}x")
    
    # Text generation samples
    print("\n" + "=" * 80)
    print("TEXT GENERATION COMPARISON")
    print("=" * 80)
    
    prompt = "To be"
    print(f"\nPrompt: '{prompt}'")
    
    for name in ['Standard', best_vsa if vsa_only else 'MAP']:
        if name in models:
            generated = generate_text(models[name], prompt, char_to_idx, idx_to_char, device)
            print(f"\n[{name}]:")
            print(f"  {generated[:100]}...")
    
    # Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    visualize_results(training_history, final_results, list(models.keys()))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: VSA TENSOR TYPE PROPERTIES")
    print("=" * 80)
    print("""
    VSA Type Properties for LLM:
    
    MAP (Multiply-Add-Permute):
    - Binding: element-wise multiply
    - Bundling: element-wise add
    - Simple, fast, robust to noise
    
    HRR (Holographic Reduced Representations):
    - Binding: circular convolution (FFT-based)
    - Bundling: element-wise add
    - Smooth similarity, good for analogies
    
    FHRR (Fourier HRR):
    - Complex-valued (phase encoding)
    - Great for resonator networks / iterative retrieval
    - May need real projection for transformers
    
    BSC (Binary Spatter Codes):
    - Binary vectors (0/1)
    - Binding: XOR, Bundling: majority
    - Memory efficient, but discrete
    
    VTB (Vector-derived Transformation Binding):
    - Matrix-based binding
    - Good for structured representations
    """)
    
    return final_results, training_history


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    results, history = main()
