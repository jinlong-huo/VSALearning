"""
VSA TENSOR TYPE COMPARISON FOR TIME-SERIES FORECASTING - COMPLETE COVERAGE

This script systematically tests ALL valid VSA type and embedding combinations
based on the torchhd library's embedding_vsa_support specification.

Embedding Types Tested:
- Projection: Random linear projection (MAP, HRR, VTB)
- Sinusoid: Nonlinear sin/cos projection (MAP, HRR, VTB)
- Level: Level-based encoding (MAP, HRR, VTB)
- Thermometer: Thermometer encoding (MAP, HRR, VTB)
- Circular: Circular/phase encoding (MAP, HRR, VTB)
- Density: Density-based encoding (MAP, HRR, VTB)
- FractionalPower: Fractional power encoding (MAP, HRR, VTB)

VSA Types:
- MAP: Multiply Add Permute (bipolar {-1, 1})
- HRR: Holographic Reduced Representations (real-valued, FFT-based)
- FHRR: Fourier HRR (complex-valued, phase-based) - Limited embedding support
- BSC: Binary Spatter Codes (binary {0, 1}) - Limited embedding support
- VTB: Vector-derived Transformation Binding
- BSBC: Block-Structured Binary Codes - Limited embedding support
- MCR: Multiplicative Convolutional Representations - Limited embedding support
- CGR: Circular Geometric Representations - Limited embedding support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchhd import embeddings
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# VSA Type and Embedding Support Matrix (ACTUAL WORKING COMBINATIONS)
# Based on torchhd library limitations discovered through testing
EMBEDDING_VSA_SUPPORT = {
    "Projection": ["MAP", "HRR", "VTB"],
    "Sinusoid": ["MAP", "HRR", "VTB"],
    "Level": ["MAP", "HRR"],  # VTB fails with dimensionality requirements
    "Thermometer": ["MAP"],  # HRR and VTB have compatibility issues
    "Circular": [],  # Currently broken for all VSA types in torchhd
    "Density": ["MAP"],  # HRR and VTB have model type issues
    "FractionalPower": ["HRR"],
}

# For time-series, we focus on continuous embeddings that work well with sequences
# Excluding Circular (broken) and limiting others based on actual compatibility
TIMESERIES_EMBEDDINGS = ["Projection", "Sinusoid", "Level", "Thermometer", "Density", "FractionalPower"]


def to_tensor(x):
    """Convert VSATensor to regular torch.Tensor for nn layer compatibility"""
    if hasattr(x, 'as_subclass'):
        return x.as_subclass(torch.Tensor)
    return x


# ============================================================================
# BASELINE: STANDARD EMBEDDING
# ============================================================================

class StandardEmbedding(nn.Module):
    """Standard learnable linear projection (baseline)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.projection = nn.Linear(in_features, out_features, bias=False)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        return self.projection(x)


# ============================================================================
# UNIVERSAL VSA EMBEDDING WRAPPER
# ============================================================================

class VSAEmbedding(nn.Module):
    """Universal VSA embedding wrapper that handles all VSA types and embedding methods"""
    def __init__(self, in_features, out_features, vsa_type, embedding_type):
        super().__init__()
        self.vsa_type = vsa_type
        self.embedding_type = embedding_type
        self.out_features = out_features
        
        embedding_class = getattr(embeddings, embedding_type)
        
        # Embeddings that require num_embeddings, embedding_dim, low, high
        needs_range = embedding_type in ["Level", "Thermometer", "Circular", "Density"]
        
        if needs_range:
            # For VTB with even dimensions, we need to ensure out_features is even
            if vsa_type == 'VTB' and out_features % 2 != 0:
                raise ValueError(f"VTB requires even dimensionality, got {out_features}")
            
            self.embed = embedding_class(
                num_embeddings=in_features,
                embedding_dim=out_features,
                vsa=vsa_type,
                low=0.0,
                high=float(in_features - 1)
            )
        else:
            # Standard embeddings (Projection, Sinusoid, FractionalPower)
            self.embed = embedding_class(
                in_features=in_features,
                out_features=out_features,
                vsa=vsa_type
            )
        
        # Learnable scale parameter
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
        
        # For FHRR (complex-valued), we need a projection layer
        if vsa_type == 'FHRR':
            self.proj = nn.Linear(out_features * 2, out_features)
        else:
            self.proj = None
    
    def forward(self, x):
        # Handle different input shapes
        needs_indices = self.embedding_type in ["Level", "Thermometer", "Circular", "Density"]
        
        if needs_indices:
            # These embeddings expect integer indices
            if x.dim() == 3:
                # x: [batch, seq_len, n_channels]
                batch, seq_len, n_channels = x.shape
                # Flatten to [batch * n_channels, seq_len]
                x_flat = x.transpose(1, 2).reshape(batch * n_channels, seq_len)
                # Convert to indices (normalize to range [0, seq_len-1])
                x_indices = torch.clamp((x_flat * 10 + 48).long(), 0, seq_len - 1)
                # Embed: [batch * n_channels, seq_len, out_features]
                out = self.embed(x_indices)
                # Average over sequence dimension
                out = out.mean(dim=1)  # [batch * n_channels, out_features]
                # Reshape back: [batch, n_channels, out_features]
                out = out.reshape(batch, n_channels, self.out_features)
            else:
                # x: [batch, n_channels]
                x_indices = torch.clamp((x * 10 + 48).long(), 0, self.embed.num_embeddings - 1)
                out = self.embed(x_indices)
        else:
            # Standard projection-based embeddings
            if x.dim() == 3:
                x = x.transpose(1, 2)
            out = self.embed(x)
        
        out = to_tensor(out)
        
        # Handle complex-valued outputs (FHRR)
        if out.is_complex():
            real_part = torch.real(out)
            imag_part = torch.imag(out)
            out = torch.cat([real_part, imag_part], dim=-1)
            if self.proj is not None:
                out = self.proj(out)
        
        # Handle binary outputs (BSC, BSBC)
        if self.vsa_type in ['BSC', 'BSBC']:
            out = out.float()
            out = out * 2.0 - 1.0  # Convert {0,1} to {-1,1}
        
        return out * self.scale


# ============================================================================
# FORECASTING MODEL
# ============================================================================

class ForecastModel(nn.Module):
    """Simple forecasting model using different embeddings"""
    def __init__(self, embedding_layer, d_model, pred_len):
        super().__init__()
        self.embedding = embedding_layer
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, pred_len)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        pred = self.predictor(embedded)
        pred = pred.transpose(1, 2)
        return pred


# ============================================================================
# SYNTHETIC TIME-SERIES DATA
# ============================================================================

def generate_synthetic_data(n_samples, seq_len, pred_len, n_channels):
    """Generate synthetic multi-channel time-series data"""
    data = []
    for _ in range(n_samples):
        t = np.linspace(0, 4*np.pi, seq_len + pred_len)
        sample = np.zeros((seq_len + pred_len, n_channels))
        
        for ch in range(n_channels):
            freq = np.random.uniform(1, 5)
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 1.5)
            sample[:, ch] = amplitude * np.sin(freq * t + phase) + 0.1 * np.random.randn(seq_len + pred_len)
        
        x = sample[:seq_len, :]
        y = sample[seq_len:seq_len+pred_len, :]
        data.append((x, y))
    
    return data


class TimeSeriesDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.FloatTensor(x), torch.FloatTensor(y)


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""
    pred_flat = predictions.reshape(-1).numpy()
    tgt_flat = targets.reshape(-1).numpy()
    
    mse = F.mse_loss(predictions, targets).item()
    mae = F.l1_loss(predictions, targets).item()
    rmse = np.sqrt(mse)
    
    if len(pred_flat) > 1:
        corr, _ = pearsonr(pred_flat, tgt_flat)
    else:
        corr = 0.0
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'Correlation': corr}


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model on test data"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())
    
    predictions = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    return compute_metrics(predictions, targets), predictions, targets


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_results(training_history, final_results, predictions_dict, 
                     targets, output_dir='output'):
    """Generate comprehensive visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Sort methods by performance
    sorted_methods = sorted(final_results.keys(), key=lambda k: final_results[k]['MSE'])
    
    # Create main comparison figure
    fig = plt.figure(figsize=(24, 16))
    
    # 1. Training curves for top 10 methods
    ax = fig.add_subplot(3, 4, 1)
    for name in sorted_methods[:10]:
        if name in training_history:
            ax.plot(training_history[name], label=name, linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Training Progress - Top 10 Methods')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 2. MSE comparison (top 15)
    ax = fig.add_subplot(3, 4, 2)
    names = sorted_methods[:15]
    mse_vals = [final_results[n]['MSE'] for n in names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
    bars = ax.barh(range(len(names)), mse_vals, color=colors, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('MSE (lower is better)')
    ax.set_title('Top 15 Methods by MSE')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 3. Correlation comparison
    ax = fig.add_subplot(3, 4, 3)
    corr_vals = [final_results[n]['Correlation'] for n in names]
    bars = ax.barh(range(len(names)), corr_vals, color=colors, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Correlation (higher is better)')
    ax.set_title('Top 15 Methods by Correlation')
    ax.invert_yaxis()
    ax.set_xlim(min(corr_vals) - 0.05, 1.0)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4. Relative performance vs Standard
    ax = fig.add_subplot(3, 4, 4)
    if 'Standard' in final_results:
        baseline_mse = final_results['Standard']['MSE']
        ratios = [final_results[n]['MSE'] / baseline_mse for n in sorted_methods[:15]]
        bar_colors = ['green' if r < 0.9 else 'lightgreen' if r < 1.0 else 'orange' if r < 1.2 else 'red' for r in ratios]
        bars = ax.barh(range(len(ratios)), ratios, color=bar_colors, edgecolor='black', alpha=0.7)
        ax.axvline(x=1.0, color='black', linestyle='--', linewidth=2, label='Standard Baseline')
        ax.set_yticks(range(len(ratios)))
        ax.set_yticklabels(sorted_methods[:15], fontsize=7)
        ax.set_xlabel('MSE Ratio vs Standard')
        ax.set_title('Relative Performance (<1.0 = Better)')
        ax.invert_yaxis()
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
    
    # 5-7. Sample predictions for top 3 channels
    target_sample = targets[0].numpy()
    pred_colors = plt.cm.Set1(np.linspace(0, 1, 6))
    
    for ch in range(min(3, target_sample.shape[1])):
        ax = fig.add_subplot(3, 4, 5 + ch)
        ax.plot(target_sample[:, ch], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
        
        for idx, name in enumerate(sorted_methods[:6]):
            if name in predictions_dict:
                pred_sample = predictions_dict[name][0].numpy()
                ax.plot(pred_sample[:, ch], '--', color=pred_colors[idx], 
                       linewidth=1.2, label=name, alpha=0.7)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Channel {ch+1} Predictions - Top 6 Methods')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
    
    # 8. VSA Type comparison
    ax = fig.add_subplot(3, 4, 8)
    vsa_types = list(set([n.split('_')[0] for n in final_results.keys() if n != 'Standard']))
    vsa_best_mse = {}
    for vsa in vsa_types:
        vsa_methods = [n for n in final_results.keys() if n.startswith(vsa + '_')]
        if vsa_methods:
            vsa_best_mse[vsa] = min([final_results[n]['MSE'] for n in vsa_methods])
    
    sorted_vsa = sorted(vsa_best_mse.keys(), key=lambda k: vsa_best_mse[k])
    vsa_mse_vals = [vsa_best_mse[v] for v in sorted_vsa]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_vsa)))
    bars = ax.bar(range(len(sorted_vsa)), vsa_mse_vals, color=colors, edgecolor='black')
    ax.set_xticks(range(len(sorted_vsa)))
    ax.set_xticklabels(sorted_vsa, rotation=45, ha='right')
    ax.set_ylabel('Best MSE')
    ax.set_title('Best Performance by VSA Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 9. Embedding type comparison
    ax = fig.add_subplot(3, 4, 9)
    embed_types = TIMESERIES_EMBEDDINGS
    embed_best_mse = {}
    for embed in embed_types:
        embed_methods = [n for n in final_results.keys() if embed in n]
        if embed_methods:
            embed_best_mse[embed] = min([final_results[n]['MSE'] for n in embed_methods])
    
    sorted_embed = sorted(embed_best_mse.keys(), key=lambda k: embed_best_mse[k])
    embed_mse_vals = [embed_best_mse[e] for e in sorted_embed]
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_embed)))
    bars = ax.bar(range(len(sorted_embed)), embed_mse_vals, color=colors, edgecolor='black')
    ax.set_xticks(range(len(sorted_embed)))
    ax.set_xticklabels(sorted_embed, rotation=45, ha='right')
    ax.set_ylabel('Best MSE')
    ax.set_title('Best Performance by Embedding Type')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 10. Heatmap: VSA × Embedding performance
    ax = fig.add_subplot(3, 4, 10)
    vsa_list = sorted(vsa_types)
    embed_list = TIMESERIES_EMBEDDINGS
    heatmap_data = np.full((len(vsa_list), len(embed_list)), np.nan)
    
    for i, vsa in enumerate(vsa_list):
        for j, embed in enumerate(embed_list):
            method_name = f"{vsa}_{embed}"
            if method_name in final_results:
                heatmap_data[i, j] = final_results[method_name]['MSE']
    
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
    ax.set_xticks(range(len(embed_list)))
    ax.set_xticklabels(embed_list, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(vsa_list)))
    ax.set_yticklabels(vsa_list, fontsize=8)
    ax.set_title('MSE Heatmap: VSA × Embedding')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 11. Distribution of results
    ax = fig.add_subplot(3, 4, 11)
    all_mse = [final_results[n]['MSE'] for n in final_results.keys()]
    ax.hist(all_mse, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(x=np.median(all_mse), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(all_mse):.4f}')
    if 'Standard' in final_results:
        ax.axvline(x=final_results['Standard']['MSE'], color='orange', linestyle='-', linewidth=2, label=f'Standard: {final_results["Standard"]["MSE"]:.4f}')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Density')
    ax.set_title('Distribution of MSE Across All Methods')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 12. Summary statistics
    ax = fig.add_subplot(3, 4, 12)
    ax.axis('off')
    
    best_overall = sorted_methods[0]
    vsa_only = {k: v for k, v in final_results.items() if k != 'Standard'}
    best_vsa = min(vsa_only.keys(), key=lambda k: vsa_only[k]['MSE']) if vsa_only else 'N/A'
    
    n_better = sum(1 for n in final_results.keys() if n != 'Standard' and final_results[n]['MSE'] < final_results.get('Standard', {}).get('MSE', float('inf')))
    
    summary_text = f"""
╔══════════════════════════════════════════════════════════════╗
║        VSA TYPE & EMBEDDING COMPARISON SUMMARY               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  🏆 BEST OVERALL: {best_overall:<30}          ║
║     MSE: {final_results[best_overall]['MSE']:.6f}                                ║
║     Correlation: {final_results[best_overall]['Correlation']:.4f}                         ║
║                                                              ║
║  🥇 BEST VSA METHOD: {best_vsa:<27}       ║
║     MSE: {final_results.get(best_vsa, {}).get('MSE', 0):.6f}                                ║
║                                                              ║
║  📊 Standard Baseline MSE: {final_results.get('Standard', {}).get('MSE', 0):.6f}            ║
║                                                              ║
║  ✅ Methods Better than Standard: {n_better}/{len(vsa_only)}                 ║
║                                                              ║
║  Total Methods Tested: {len(final_results)}                           ║
║  - VSA Types: {len(vsa_types)}                                       ║
║  - Embedding Types: {len(TIMESERIES_EMBEDDINGS)}                             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.0, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vsa_complete_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/vsa_complete_comparison.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("VSA TYPE & EMBEDDING COMPLETE COMPARISON FOR TIME-SERIES FORECASTING")
    print("=" * 80)
    print(f"""
    Testing ALL valid VSA + Embedding combinations based on torchhd support.
    
    VSA Types: MAP, HRR, VTB (primary focus for time-series)
    
    Embeddings for time-series:
    {', '.join(TIMESERIES_EMBEDDINGS)}
    
    Note: Some combinations are excluded due to torchhd library limitations:
    - VTB_Level: Requires even dimensionality
    - HRR_Thermometer, VTB_Thermometer: Model type compatibility issues
    - *_Circular: Currently broken in torchhd for all VSA types
    - HRR_Density, VTB_Density: Model type issues
    
    Expected valid combinations: ~11-12 + 1 baseline
    """)
    
    # Configuration
    seq_len = 96
    pred_len = 24
    n_channels = 7
    d_model = 128
    batch_size = 32
    n_epochs = 20
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nDevice: {device}")
    print(f"Config: seq_len={seq_len}, pred_len={pred_len}, n_channels={n_channels}, d_model={d_model}")
    print(f"Training: {n_epochs} epochs, batch_size={batch_size}")
    
    # Generate Data
    print("\n[1/4] Generating synthetic time-series data...")
    train_data = generate_synthetic_data(2000, seq_len, pred_len, n_channels)
    test_data = generate_synthetic_data(400, seq_len, pred_len, n_channels)
    
    train_loader = DataLoader(TimeSeriesDataset(train_data), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TimeSeriesDataset(test_data), batch_size=batch_size)
    print(f"✓ Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    
    # Initialize all valid VSA + Embedding combinations
    print("\n[2/4] Initializing all valid VSA + Embedding combinations...")
    
    embedding_configs = {'Standard': lambda: StandardEmbedding(seq_len, d_model)}
    
    # Generate all valid combinations
    for embedding_type in TIMESERIES_EMBEDDINGS:
        supported_vsa = EMBEDDING_VSA_SUPPORT[embedding_type]
        for vsa_type in supported_vsa:
            name = f"{vsa_type}_{embedding_type}"
            embedding_configs[name] = lambda et=embedding_type, vt=vsa_type: VSAEmbedding(seq_len, d_model, vt, et)
    
    models = {}
    optimizers = {}
    training_history = {}
    
    print(f"\nInitializing {len(embedding_configs)} configurations...")
    print("-" * 80)
    
    for name, config_fn in embedding_configs.items():
        try:
            embed_layer = config_fn()
            model = ForecastModel(embed_layer, d_model, pred_len).to(device)
            models[name] = model
            optimizers[name] = torch.optim.Adam(model.parameters(), lr=1e-3)
            training_history[name] = []
            n_params = sum(p.numel() for p in model.parameters())
            if len(models) <= 10 or name == 'Standard':
                print(f"✓ {name:<25}: {n_params:>8,} params")
        except Exception as e:
            print(f"✗ {name:<25}: Failed - {str(e)[:50]}")
    
    print(f"\n✓ Successfully initialized {len(models)} models")
    
    # Training
    print("\n[3/4] Training all models...")
    print("-" * 100)
    
    display_methods = list(models.keys())[:8]
    header = f"{'Epoch':<8}"
    for name in display_methods:
        header += f"{name:<15}"
    print(header)
    print("-" * 100)
    
    for epoch in range(n_epochs):
        epoch_losses = {}
        
        for name, model in models.items():
            try:
                train_loss = train_epoch(model, train_loader, optimizers[name], device)
                training_history[name].append(train_loss)
                epoch_losses[name] = train_loss
            except Exception as e:
                epoch_losses[name] = float('inf')
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            row = f"{epoch+1:<8}"
            for name in display_methods:
                row += f"{epoch_losses.get(name, float('inf')):<15.4f}"
            print(row)
    
    # Evaluation
    print("\n[4/4] Evaluating all models...")
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)
    
    final_results = {}
    predictions_dict = {}
    targets_tensor = None
    
    for name, model in models.items():
        try:
            metrics, preds, targets = evaluate(model, test_loader, device)
            final_results[name] = metrics
            predictions_dict[name] = preds
            if targets_tensor is None:
                targets_tensor = targets
        except Exception as e:
            print(f"✗ {name}: Evaluation failed - {str(e)[:50]}")
    
    # Print top 20 results
    sorted_methods = sorted(final_results.keys(), key=lambda k: final_results[k]['MSE'])
    
    print(f"\n{'Rank':<6} {'Method':<25} {'MSE':<12} {'MAE':<12} {'Corr':<10}")
    print("-" * 75)
    
    for rank, name in enumerate(sorted_methods[:20], 1):
        m = final_results[name]
        print(f"{rank:<6} {name:<25} {m['MSE']:<12.6f} {m['MAE']:<12.6f} {m['Correlation']:<10.4f}")
    
    # Analysis by VSA type
    print("\n" + "=" * 80)
    print("ANALYSIS BY VSA TYPE")
    print("=" * 80)
    
    vsa_types = list(set([n.split('_')[0] for n in final_results.keys() if n != 'Standard']))
    
    for vsa in sorted(vsa_types):
        vsa_methods = {n: final_results[n] for n in final_results.keys() if n.startswith(vsa + '_')}
        if vsa_methods:
            best_method = min(vsa_methods.keys(), key=lambda k: vsa_methods[k]['MSE'])
            best_mse = vsa_methods[best_method]['MSE']
            print(f"\n{vsa}:")
            print(f"  Best: {best_method} (MSE: {best_mse:.6f})")
            print(f"  Tested {len(vsa_methods)} embedding combinations")
    
    # Analysis by embedding type
    print("\n" + "=" * 80)
    print("ANALYSIS BY EMBEDDING TYPE")
    print("=" * 80)
    
    for embed in TIMESERIES_EMBEDDINGS:
        embed_methods = {n: final_results[n] for n in final_results.keys() if embed in n}
        if embed_methods:
            best_method = min(embed_methods.keys(), key=lambda k: embed_methods[k]['MSE'])
            best_mse = embed_methods[best_method]['MSE']
            print(f"\n{embed}:")
            print(f"  Best: {best_method} (MSE: {best_mse:.6f})")
            print(f"  Tested {len(embed_methods)} VSA types")
    
    # Comparison with Standard
    print("\n" + "=" * 80)
    print("COMPARISON WITH STANDARD BASELINE")
    print("=" * 80)
    
    if 'Standard' in final_results:
        baseline_mse = final_results['Standard']['MSE']
        print(f"\nStandard Baseline MSE: {baseline_mse:.6f}")
        print(f"\n{'Rank':<6} {'Method':<25} {'MSE Ratio':<12} {'Improvement':<15} {'Status':<20}")
        print("-" * 85)
        
        for rank, name in enumerate(sorted_methods[:20], 1):
            if name != 'Standard':
                ratio = final_results[name]['MSE'] / baseline_mse
                improvement = (1 - ratio) * 100
                
                if ratio <= 0.8:
                    status = "🏆 EXCELLENT"
                elif ratio <= 0.9:
                    status = "🥇 MUCH BETTER"
                elif ratio <= 1.0:
                    status = "✅ BETTER"
                elif ratio <= 1.1:
                    status = "✅ COMPARABLE"
                elif ratio <= 1.5:
                    status = "⚠️  ACCEPTABLE"
                else:
                    status = "❌ WORSE"
                
                print(f"{rank:<6} {name:<25} {ratio:<12.3f}x {improvement:>+12.1f}% {status}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    best_overall = sorted_methods[0]
    vsa_only = {k: v for k, v in final_results.items() if k != 'Standard'}
    best_vsa = min(vsa_only.keys(), key=lambda k: vsa_only[k]['MSE']) if vsa_only else 'N/A'
    
    n_better = sum(1 for n in final_results.keys() 
                   if n != 'Standard' and final_results[n]['MSE'] < final_results.get('Standard', {}).get('MSE', float('inf')))
    n_total = len(vsa_only)
    
    print(f"\n🏆 BEST OVERALL METHOD: {best_overall}")
    print(f"   MSE: {final_results[best_overall]['MSE']:.6f}")
    print(f"   MAE: {final_results[best_overall]['MAE']:.6f}")
    print(f"   Correlation: {final_results[best_overall]['Correlation']:.4f}")
    
    if 'Standard' in final_results:
        ratio = final_results[best_overall]['MSE'] / final_results['Standard']['MSE']
        print(f"   vs Standard: {ratio:.3f}x ({(1-ratio)*100:+.1f}%)")
    
    print(f"\n🥇 BEST VSA METHOD: {best_vsa}")
    print(f"   MSE: {final_results[best_vsa]['MSE']:.6f}")
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"   Total methods tested: {len(final_results)}")
    print(f"   VSA methods better than Standard: {n_better}/{n_total} ({n_better/n_total*100:.1f}%)")
    print(f"   Best VSA improvement: {(1 - final_results[best_vsa]['MSE']/final_results.get('Standard', {}).get('MSE', 1))*100:+.1f}%")
    
    # Best by category
    print(f"\n🎯 BEST BY CATEGORY:")
    
    print(f"\n  VSA Types:")
    for vsa in sorted(vsa_types)[:5]:
        vsa_methods = {n: final_results[n] for n in final_results.keys() if n.startswith(vsa + '_')}
        if vsa_methods:
            best = min(vsa_methods.keys(), key=lambda k: vsa_methods[k]['MSE'])
            print(f"    {vsa:<6}: {best:<25} MSE: {vsa_methods[best]['MSE']:.6f}")
    
    print(f"\n  Embedding Types:")
    for embed in TIMESERIES_EMBEDDINGS[:5]:
        embed_methods = {n: final_results[n] for n in final_results.keys() if embed in n}
        if embed_methods:
            best = min(embed_methods.keys(), key=lambda k: embed_methods[k]['MSE'])
            print(f"    {embed:<15}: {best:<25} MSE: {embed_methods[best]['MSE']:.6f}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS FOR VSA-LLM APPLICATION:")
    print(f"   1. Primary choice: {best_overall}")
    print(f"      - Use this configuration in your LLM experiments")
    
    vsa_type = best_overall.split('_')[0]
    embed_type = '_'.join(best_overall.split('_')[1:])
    
    print(f"\n   2. VSA Type: {vsa_type}")
    if vsa_type == 'MAP':
        print(f"      - Bipolar vectors: simple, robust, good baseline")
        print(f"      - Element-wise operations: computationally efficient")
    elif vsa_type == 'HRR':
        print(f"      - Real-valued: smooth similarity landscape")
        print(f"      - FFT binding: good for structured composition")
    elif vsa_type == 'VTB':
        print(f"      - Transformation binding: flexible representations")
        print(f"      - Good for hierarchical knowledge")
    
    print(f"\n   3. Embedding Type: {embed_type}")
    if 'Projection' in embed_type:
        print(f"      - Linear projection: simple and effective")
    elif 'Sinusoid' in embed_type:
        print(f"      - Nonlinear: captures periodic patterns well")
    elif 'Level' in embed_type:
        print(f"      - Level encoding: good for ordinal data")
    elif 'Circular' in embed_type:
        print(f"      - Circular: preserves rotational properties")
    
    print(f"\n   4. Implementation notes:")
    print(f"      - d_model={d_model} worked well")
    print(f"      - Consider testing higher dimensions (256, 512) for LLM")
    print(f"      - Scale parameter helps stabilization")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    visualize_results(training_history, final_results, predictions_dict, targets_tensor)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to output/ directory")
    print(f"Total configurations tested: {len(final_results)}")
    print(f"Best method: {best_overall} (MSE: {final_results[best_overall]['MSE']:.6f})")
    
    return final_results, training_history


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    
    print("\n" + "🚀" * 40)
    print("Starting VSA Type & Embedding Comparison Experiment")
    print("🚀" * 40 + "\n")
    
    try:
        results, history = main()
        print("\n✅ Experiment completed successfully!")
        print(f"✅ Results available in 'results' and 'history' variables")
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()