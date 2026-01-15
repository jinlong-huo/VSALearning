"""
VSA TENSOR TYPE COMPARISON FOR TIME-SERIES FORECASTING

This script compares different VSA TENSOR TYPES (architectures) on time-series
forecasting tasks to determine which VSA type works best.

VSA Tensor Types Tested:
- MAP: Multiply Add Permute (dense bipolar {-1, 1})
- HRR: Holographic Reduced Representations (real-valued, FFT-based binding)
- FHRR: Fourier HRR (complex-valued, phase-based)
- BSC: Binary Spatter Codes (binary {0, 1})
- VTB: Vector-derived Transformation Binding

Each VSA type is tested with both:
1. Projection embedding (random projection)
2. Sinusoid embedding (nonlinear sin/cos projection)

The goal is to find the optimal VSA architecture for time-series tasks
before applying it to LLM applications.
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
        # x: [batch, seq_len, n_channels]
        if x.dim() == 3:
            x = x.transpose(1, 2)  # -> [batch, n_channels, seq_len]
        return self.projection(x)


# ============================================================================
# VSA TYPE EMBEDDINGS - PROJECTION BASED
# ============================================================================

class VSA_MAP_Projection(nn.Module):
    """
    MAP: Multiply Add Permute with Projection
    - Dense bipolar vectors {-1, 1}
    - Binding: element-wise multiplication
    - Bundling: element-wise addition
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Projection(
            in_features=in_features,
            out_features=out_features,
            vsa='MAP'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_HRR_Projection(nn.Module):
    """
    HRR: Holographic Reduced Representations with Projection
    - Real-valued continuous vectors
    - Binding: circular convolution (FFT-based)
    - Bundling: element-wise addition
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Projection(
            in_features=in_features,
            out_features=out_features,
            vsa='HRR'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_FHRR_Projection(nn.Module):
    """
    FHRR: Fourier HRR with Projection
    - Complex-valued vectors (phase-based encoding)
    - Binding: element-wise complex multiplication
    - Outputs complex, needs conversion to real
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.embed = embeddings.Projection(
            in_features=in_features,
            out_features=out_features,
            vsa='FHRR'
        )
        # Project complex to real
        self.proj = nn.Linear(out_features * 2, out_features)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        out = to_tensor(out)
        
        # Convert complex to real by concatenating real and imaginary parts
        if out.is_complex():
            real_part = torch.real(out)
            imag_part = torch.imag(out)
            out = torch.cat([real_part, imag_part], dim=-1)
            out = self.proj(out)
        
        return out * self.scale


class VSA_BSC_Projection(nn.Module):
    """
    BSC: Binary Spatter Codes with Projection
    - Binary vectors {0, 1}
    - Binding: XOR operation
    - Bundling: majority vote
    - Convert binary to float for gradient flow
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Projection(
            in_features=in_features,
            out_features=out_features,
            vsa='BSC'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        out = to_tensor(out).float()
        # Convert {0, 1} to {-1, 1} for better gradient flow
        out = out * 2.0 - 1.0
        return out * self.scale


class VSA_VTB_Projection(nn.Module):
    """
    VTB: Vector-derived Transformation Binding with Projection
    - Real-valued vectors
    - Binding: matrix transformation derived from vectors
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Projection(
            in_features=in_features,
            out_features=out_features,
            vsa='VTB'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


# ============================================================================
# VSA TYPE EMBEDDINGS - SINUSOID BASED
# ============================================================================

class VSA_MAP_Sinusoid(nn.Module):
    """MAP with Sinusoid nonlinear projection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Sinusoid(
            in_features=in_features,
            out_features=out_features,
            vsa='MAP'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_HRR_Sinusoid(nn.Module):
    """HRR with Sinusoid nonlinear projection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Sinusoid(
            in_features=in_features,
            out_features=out_features,
            vsa='HRR'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


class VSA_FHRR_Sinusoid(nn.Module):
    """FHRR with Sinusoid nonlinear projection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features
        self.embed = embeddings.Sinusoid(
            in_features=in_features,
            out_features=out_features,
            vsa='FHRR'
        )
        self.proj = nn.Linear(out_features * 2, out_features)
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        out = to_tensor(out)
        
        if out.is_complex():
            real_part = torch.real(out)
            imag_part = torch.imag(out)
            out = torch.cat([real_part, imag_part], dim=-1)
            out = self.proj(out)
        
        return out * self.scale


class VSA_BSC_Sinusoid(nn.Module):
    """BSC with Sinusoid nonlinear projection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Sinusoid(
            in_features=in_features,
            out_features=out_features,
            vsa='BSC'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        out = to_tensor(out).float()
        out = out * 2.0 - 1.0
        return out * self.scale


class VSA_VTB_Sinusoid(nn.Module):
    """VTB with Sinusoid nonlinear projection"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.embed = embeddings.Sinusoid(
            in_features=in_features,
            out_features=out_features,
            vsa='VTB'
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.transpose(1, 2)
        out = self.embed(x)
        return to_tensor(out) * self.scale


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
        # x: [batch, seq_len, n_channels]
        embedded = self.embedding(x)  # -> [batch, n_channels, d_model]
        pred = self.predictor(embedded)  # -> [batch, n_channels, pred_len]
        pred = pred.transpose(1, 2)  # -> [batch, pred_len, n_channels]
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
            # Sinusoidal pattern with noise
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

def visualize_results(training_history, final_results, method_names, predictions_dict, 
                     targets, output_dir='output'):
    """Generate comprehensive visualizations"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 16))
    
    # =========================================================================
    # Plot 1: Training Curves
    # =========================================================================
    ax = fig.add_subplot(3, 3, 1)
    for name in method_names:
        if name in training_history:
            ax.plot(training_history[name], label=name, linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.set_title('Training Progress by VSA Type')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # =========================================================================
    # Plot 2: Final MSE Comparison (Bar Chart)
    # =========================================================================
    ax = fig.add_subplot(3, 3, 2)
    names = [n for n in method_names if n in final_results]
    mse_vals = [final_results[n]['MSE'] for n in names]
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
    bars = ax.bar(range(len(names)), mse_vals, color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MSE (lower is better)')
    ax.set_title('Final Test MSE by VSA Type')
    
    for bar, val in zip(bars, mse_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    
    # =========================================================================
    # Plot 3: Correlation Comparison
    # =========================================================================
    ax = fig.add_subplot(3, 3, 3)
    corr_vals = [final_results[n]['Correlation'] for n in names]
    bars = ax.bar(range(len(names)), corr_vals, color=colors, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Correlation (higher is better)')
    ax.set_title('Prediction Correlation by VSA Type')
    ax.set_ylim(min(corr_vals) - 0.05, 1.0)
    
    for bar, val in zip(bars, corr_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
               f'{val:.4f}', ha='center', va='bottom', fontsize=7)
    
    # =========================================================================
    # Plot 4-6: Prediction Samples (3 channels)
    # =========================================================================
    target_sample = targets[0].numpy()  # First sample
    
    # Select top 6 methods for prediction visualization
    sorted_methods = sorted(final_results.keys(), key=lambda k: final_results[k]['MSE'])[:6]
    pred_colors = plt.cm.Set1(np.linspace(0, 1, len(sorted_methods)))
    
    for ch in range(min(3, target_sample.shape[1])):
        ax = fig.add_subplot(3, 3, 4 + ch)
        ax.plot(target_sample[:, ch], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.9)
        
        for idx, name in enumerate(sorted_methods):
            if name in predictions_dict:
                pred_sample = predictions_dict[name][0].numpy()
                ax.plot(pred_sample[:, ch], '--', color=pred_colors[idx], 
                       linewidth=1.2, label=name, alpha=0.7)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Channel {ch+1} Predictions')
        ax.legend(fontsize=6, loc='best')
        ax.grid(True, alpha=0.3)
    
    # =========================================================================
    # Plot 7: Relative Performance vs Standard
    # =========================================================================
    ax = fig.add_subplot(3, 3, 7)
    if 'Standard' in final_results:
        baseline_mse = final_results['Standard']['MSE']
        ratios = [final_results[n]['MSE'] / baseline_mse for n in names]
        
        bar_colors = ['green' if r <= 1.0 else 'orange' if r <= 1.2 else 'red' for r in ratios]
        bars = ax.bar(range(len(names)), ratios, color=bar_colors, edgecolor='black', alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Standard Baseline')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('MSE Ratio vs Standard')
        ax.set_title('Relative Performance (< 1.0 = Better than Standard)')
        ax.legend()
    
    # =========================================================================
    # Plot 8: VSA Type Grouping (Projection vs Sinusoid)
    # =========================================================================
    ax = fig.add_subplot(3, 3, 8)
    
    # Group by VSA type
    vsa_types = ['MAP', 'HRR', 'FHRR', 'BSC', 'VTB']
    proj_mse = []
    sin_mse = []
    
    for vsa in vsa_types:
        proj_name = f'{vsa}_Proj'
        sin_name = f'{vsa}_Sin'
        proj_mse.append(final_results.get(proj_name, {}).get('MSE', np.nan))
        sin_mse.append(final_results.get(sin_name, {}).get('MSE', np.nan))
    
    x = np.arange(len(vsa_types))
    width = 0.35
    
    ax.bar(x - width/2, proj_mse, width, label='Projection', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, sin_mse, width, label='Sinusoid', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(vsa_types)
    ax.set_ylabel('MSE')
    ax.set_title('Projection vs Sinusoid by VSA Type')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Plot 9: Summary Table
    # =========================================================================
    ax = fig.add_subplot(3, 3, 9)
    ax.axis('off')
    
    # Find best methods
    best_overall = min(final_results.keys(), key=lambda k: final_results[k]['MSE'])
    vsa_only = {k: v for k, v in final_results.items() if k != 'Standard'}
    best_vsa = min(vsa_only.keys(), key=lambda k: vsa_only[k]['MSE']) if vsa_only else 'N/A'
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║             VSA TYPE COMPARISON SUMMARY                      ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║  🏆 BEST OVERALL: {best_overall:<20}                   ║
    ║     MSE: {final_results[best_overall]['MSE']:.6f}                                    ║
    ║     Correlation: {final_results[best_overall]['Correlation']:.4f}                             ║
    ║                                                              ║
    ║  🥇 BEST VSA TYPE: {best_vsa:<20}                  ║
    ║     MSE: {final_results.get(best_vsa, {}).get('MSE', 0):.6f}                                    ║
    ║                                                              ║
    ║  📊 Standard Baseline MSE: {final_results.get('Standard', {}).get('MSE', 0):.6f}                ║
    ║                                                              ║
    ║  VSA Types Tested: MAP, HRR, FHRR, BSC, VTB                  ║
    ║  Embedding Methods: Projection, Sinusoid                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.0, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vsa_type_timeseries_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/vsa_type_timeseries_comparison.png")
    plt.close()
    
    # =========================================================================
    # Additional: Embedding Heatmaps
    # =========================================================================
    fig2, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig2.suptitle('Embedding Space Visualization by VSA Type', fontsize=14)
    
    # This would require storing embeddings during evaluation
    # For now, create a placeholder
    for idx, name in enumerate(sorted_methods[:10]):
        row = idx // 5
        col = idx % 5
        if row < 2 and col < 5:
            ax = axes[row, col]
            ax.text(0.5, 0.5, name, ha='center', va='center', fontsize=12,
                   transform=ax.transAxes)
            ax.set_title(f'MSE: {final_results.get(name, {}).get("MSE", 0):.4f}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vsa_type_embeddings.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/vsa_type_embeddings.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("VSA TENSOR TYPE COMPARISON FOR TIME-SERIES FORECASTING")
    print("=" * 80)
    print("""
    Testing different VSA ARCHITECTURES:
    
    VSA Types:
    - MAP: Multiply Add Permute (bipolar {-1,1})
    - HRR: Holographic Reduced Representations (real, FFT binding)
    - FHRR: Fourier HRR (complex, phase encoding)
    - BSC: Binary Spatter Codes (binary {0,1})
    - VTB: Vector-derived Transformation Binding
    
    Embedding Methods for each VSA type:
    - Projection: Random linear projection
    - Sinusoid: Nonlinear sin/cos projection
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
    
    # Initialize all VSA type configurations
    print("\n[2/4] Initializing VSA type embeddings...")
    
    embedding_configs = {
        # Baseline
        'Standard': lambda: StandardEmbedding(seq_len, d_model),
        
        # MAP VSA
        'MAP_Proj': lambda: VSA_MAP_Projection(seq_len, d_model),
        'MAP_Sin': lambda: VSA_MAP_Sinusoid(seq_len, d_model),
        
        # HRR VSA
        'HRR_Proj': lambda: VSA_HRR_Projection(seq_len, d_model),
        'HRR_Sin': lambda: VSA_HRR_Sinusoid(seq_len, d_model),
        
        # FHRR VSA
        'FHRR_Proj': lambda: VSA_FHRR_Projection(seq_len, d_model),
        'FHRR_Sin': lambda: VSA_FHRR_Sinusoid(seq_len, d_model),
        
        # BSC VSA
        'BSC_Proj': lambda: VSA_BSC_Projection(seq_len, d_model),
        'BSC_Sin': lambda: VSA_BSC_Sinusoid(seq_len, d_model),
        
        # VTB VSA
        'VTB_Proj': lambda: VSA_VTB_Projection(seq_len, d_model),
        'VTB_Sin': lambda: VSA_VTB_Sinusoid(seq_len, d_model),
    }
    
    models = {}
    optimizers = {}
    training_history = {}
    
    for name, config_fn in embedding_configs.items():
        try:
            embed_layer = config_fn()
            model = ForecastModel(embed_layer, d_model, pred_len).to(device)
            models[name] = model
            optimizers[name] = torch.optim.Adam(model.parameters(), lr=1e-3)
            training_history[name] = []
            n_params = sum(p.numel() for p in model.parameters())
            print(f"✓ {name:<12}: {n_params:>8,} params")
        except Exception as e:
            print(f"✗ {name:<12}: Failed - {e}")
    
    # Training
    print("\n[3/4] Training all models...")
    print("-" * 100)
    
    header = f"{'Epoch':<8}"
    for name in list(models.keys())[:8]:  # Show first 8
        header += f"{name:<12}"
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
        
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            row = f"{epoch+1:<8}"
            for name in list(models.keys())[:8]:
                row += f"{epoch_losses.get(name, float('inf')):<12.4f}"
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
            print(f"✗ {name}: Evaluation failed - {e}")
    
    # Print results table
    print(f"\n{'Method':<12} {'MSE':<12} {'MAE':<12} {'RMSE':<12} {'Corr':<12}")
    print("-" * 60)
    
    for name in models.keys():
        if name in final_results:
            m = final_results[name]
            print(f"{name:<12} {m['MSE']:<12.6f} {m['MAE']:<12.6f} {m['RMSE']:<12.6f} {m['Correlation']:<12.4f}")
    
    # Comparison with Standard
    print("\n" + "=" * 80)
    print("COMPARISON WITH STANDARD BASELINE")
    print("=" * 80)
    
    baseline_mse = final_results.get('Standard', {}).get('MSE', 1.0)
    print(f"\n{'Method':<12} {'MSE Ratio':<12} {'Improvement':<15} {'Status':<20}")
    print("-" * 65)
    
    for name in models.keys():
        if name in final_results:
            ratio = final_results[name]['MSE'] / baseline_mse
            improvement = (1 - ratio) * 100
            
            if ratio <= 0.9:
                status = "🏆 MUCH BETTER"
            elif ratio <= 1.0:
                status = "✅ BETTER"
            elif ratio <= 1.1:
                status = "✅ COMPARABLE"
            elif ratio <= 1.5:
                status = "⚠️  ACCEPTABLE"
            else:
                status = "❌ WORSE"
            
            print(f"{name:<12} {ratio:<12.3f}x {improvement:>+12.1f}% {status}")
    
    # Best VSA type summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    vsa_only = {k: v for k, v in final_results.items() if k != 'Standard'}
    if vsa_only:
        best_vsa = min(vsa_only.keys(), key=lambda k: vsa_only[k]['MSE'])
        best_overall = min(final_results.keys(), key=lambda k: final_results[k]['MSE'])
        
        print(f"\n🏆 BEST VSA TYPE: {best_vsa}")
        print(f"   MSE: {vsa_only[best_vsa]['MSE']:.6f}")
        print(f"   Correlation: {vsa_only[best_vsa]['Correlation']:.4f}")
        print(f"   vs Standard: {vsa_only[best_vsa]['MSE']/baseline_mse:.3f}x ({(1-vsa_only[best_vsa]['MSE']/baseline_mse)*100:+.1f}%)")
        
        print(f"\n🎯 BEST OVERALL: {best_overall}")
        print(f"   MSE: {final_results[best_overall]['MSE']:.6f}")
        
        # Group analysis
        print("\n📊 VSA TYPE RANKING (by best performer in each type):")
        vsa_types = ['MAP', 'HRR', 'FHRR', 'BSC', 'VTB']
        type_best = {}
        
        for vsa in vsa_types:
            proj_name = f'{vsa}_Proj'
            sin_name = f'{vsa}_Sin'
            proj_mse = final_results.get(proj_name, {}).get('MSE', float('inf'))
            sin_mse = final_results.get(sin_name, {}).get('MSE', float('inf'))
            best_mse = min(proj_mse, sin_mse)
            best_method = proj_name if proj_mse <= sin_mse else sin_name
            type_best[vsa] = {'mse': best_mse, 'method': best_method}
        
        sorted_types = sorted(type_best.keys(), key=lambda k: type_best[k]['mse'])
        for rank, vsa in enumerate(sorted_types, 1):
            info = type_best[vsa]
            print(f"   {rank}. {vsa:<6} - MSE: {info['mse']:.6f} ({info['method']})")
    
    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    visualize_results(training_history, final_results, list(models.keys()), 
                     predictions_dict, targets_tensor)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR VSA-LLM")
    print("=" * 80)
    print("""
    Based on time-series results, consider for LLM:
    
    1. If best VSA type beats Standard:
       → Use that VSA type in vsa_llm_train.py
       
    2. Sinusoid vs Projection:
       → Choose based on which performs better
       
    3. Properties to consider for LLM:
       - MAP: Simple, robust, good baseline
       - HRR: Smooth similarity, good for analogies
       - FHRR: Complex ops, may need more tuning
       - BSC: Memory efficient but discrete
       - VTB: Good for structured knowledge
    """)
    
    return final_results, training_history


if __name__ == "__main__":
    import os
    os.makedirs('output', exist_ok=True)
    results, history = main()
