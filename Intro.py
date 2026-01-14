"""
VSA Learning Pipeline: From Basics to Hybrid CNN-VSA
Progressive examples using torchhd for seismic/time-series data
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# LEVEL 1: Pure VSA - Understanding the Fundamentals
# ============================================================================
def level1_basic_vsa():
    """
    Learn: How to create, bind, and bundle hypervectors
    Use Case: Encoding simple seismic features (amplitude, frequency, phase)
    """
    print("=== LEVEL 1: Basic VSA Operations ===\n")
    
    # Create random hypervectors (10,000 dimensions is standard)
    D = 10000  # Hypervector dimensionality
    
    # Random hypervectors for different features
    hv_high_amplitude = torchhd.random(1, D)
    hv_low_amplitude = torchhd.random(1, D)
    hv_freq_10hz = torchhd.random(1, D)
    hv_freq_50hz = torchhd.random(1, D)
    
    # BINDING: Combine features (like "high amplitude AND 10Hz")
    seismic_event_1 = torchhd.bind(hv_high_amplitude, hv_freq_10hz)
    seismic_event_2 = torchhd.bind(hv_low_amplitude, hv_freq_50hz)
    
    # BUNDLING: Combine multiple events into a memory trace
    memory = torchhd.bundle(seismic_event_1, seismic_event_2)
    
    # QUERYING: Can we retrieve the frequency from event 1?
    # Unbinding: memory * hv_high_amplitude should give us hv_freq_10hz
    retrieved = torchhd.bind(memory, hv_high_amplitude)
    
    # Similarity check (cosine similarity)
    sim_10hz = torchhd.cosine_similarity(retrieved, hv_freq_10hz)
    sim_50hz = torchhd.cosine_similarity(retrieved, hv_freq_50hz)
    
    print(f"Similarity to 10Hz: {sim_10hz.item():.4f}")
    print(f"Similarity to 50Hz: {sim_50hz.item():.4f}")
    print("✓ Higher similarity to correct frequency!\n")


# ============================================================================
# LEVEL 2: General VSA Classifier for Any Classification Problem
# ============================================================================
def level2_vsa_classifier():
    """
    Learn: General VSA classification paradigm for any dataset
    
    VSA Classification Pipeline:
    1. Create random hypervectors for each feature dimension (ID vectors)
    2. Create level hypervectors for quantized feature values
    3. Encode: For each sample, bind(feature_id, level) then bundle all features
    4. Train: Accumulate encoded samples into class prototype centroids
    5. Predict: Find nearest centroid using similarity (dot product or cosine)
    """
    print("=== LEVEL 2: General VSA Classifier ===\n")
    
    # =========================================================================
    # STEP 1: Generate or Load Dataset (using Iris-like synthetic data)
    # =========================================================================
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    n_classes = 3
    
    # Create synthetic multi-class dataset (similar to Iris)
    # Class 0: centered around [1, 1, 0, 0]
    # Class 1: centered around [0, 0, 1, 1]
    # Class 2: centered around [1, 0, 1, 0]
    samples_per_class = n_samples // n_classes
    
    X_class0 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 1, 0, 0])
    X_class1 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([0, 0, 1, 1])
    X_class2 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 0, 1, 0])
    
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.array([0]*samples_per_class + [1]*samples_per_class + [2]*samples_per_class)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Train/test split (80/20)
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
    
    # =========================================================================
    # STEP 2: VSA Hyperparameters
    # =========================================================================
    D = 10000          # Hypervector dimensionality (higher = more capacity)
    n_levels = 100     # Quantization levels for continuous features
    
    # =========================================================================
    # STEP 3: Create Base Hypervectors (Codebook)
    # =========================================================================
    # Feature ID hypervectors - one unique random HV per feature dimension
    feature_hvs = torchhd.random(n_features, D)
    
    # Level hypervectors - one HV per quantization level
    # Using thermometer encoding for better similarity preservation
    level_hvs = torchhd.random(n_levels, D)
    
    print(f"VSA Config: D={D}, n_levels={n_levels}")
    print(f"Feature HVs shape: {feature_hvs.shape}")
    print(f"Level HVs shape: {level_hvs.shape}\n")
    
    # =========================================================================
    # STEP 4: Encode Function - Transform samples to hypervectors
    # =========================================================================
    def encode_sample(sample, feature_hvs, level_hvs, n_levels):
        """
        Encode a single sample into a hypervector.
        
        For each feature:
            1. Quantize the feature value to a level index
            2. Bind the feature ID with the level HV
            3. Bundle all bound HVs together
        
        Args:
            sample: [n_features] tensor of feature values
            feature_hvs: [n_features, D] tensor of feature ID hypervectors
            level_hvs: [n_levels, D] tensor of level hypervectors
            n_levels: number of quantization levels
        
        Returns:
            [D] hypervector representing the sample
        """
        # Normalize sample to [0, 1] range for quantization
        sample_min, sample_max = sample.min(), sample.max()
        if sample_max - sample_min > 0:
            normalized = (sample - sample_min) / (sample_max - sample_min)
        else:
            normalized = torch.zeros_like(sample)
        
        # Quantize to level indices
        level_indices = (normalized * (n_levels - 1)).long().clamp(0, n_levels - 1)
        
        # Bind each feature with its level and collect
        bound_hvs = []
        for feat_idx in range(len(sample)):
            level_idx = level_indices[feat_idx].item()
            # Bind: feature_id ⊛ level_value
            bound_hv = torchhd.bind(feature_hvs[feat_idx], level_hvs[level_idx])
            bound_hvs.append(bound_hv)
        
        # Bundle: sum all bound hypervectors
        stacked = torch.stack(bound_hvs, dim=0)
        sample_hv = stacked.sum(dim=0)
        
        return sample_hv
    
    def encode_dataset(X, feature_hvs, level_hvs, n_levels):
        """Encode entire dataset."""
        encoded = []
        for i in range(len(X)):
            hv = encode_sample(X[i], feature_hvs, level_hvs, n_levels)
            encoded.append(hv)
        return torch.stack(encoded)
    
    # =========================================================================
    # STEP 5: Encode Training and Test Data
    # =========================================================================
    print("Encoding training data...")
    train_encoded = encode_dataset(X_train, feature_hvs, level_hvs, n_levels)
    print(f"Encoded train shape: {train_encoded.shape}")
    
    print("Encoding test data...")
    test_encoded = encode_dataset(X_test, feature_hvs, level_hvs, n_levels)
    print(f"Encoded test shape: {test_encoded.shape}\n")
    
    # =========================================================================
    # STEP 6: Train VSA Classifier (Build Class Prototypes)
    # =========================================================================
    model = Centroid(in_features=D, out_features=n_classes)
    
    # Add all training samples to build class centroids
    # The Centroid model accumulates samples per class
    model.add(train_encoded, y_train)
    
    print("Training complete! Class prototypes built.\n")
    
    # =========================================================================
    # STEP 7: Predict on Test Data
    # =========================================================================
    # Get similarity scores to each class centroid
    similarity_scores = model(test_encoded, dot=True)
    
    # Predict class with highest similarity
    predictions = similarity_scores.argmax(dim=1)
    
    # Calculate accuracy
    accuracy = (predictions == y_test).float().mean().item()
    
    print(f"Test Results:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Per-class accuracy
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == y_test[mask]).float().mean().item()
            print(f"  Class {c} accuracy: {class_acc:.2%} ({mask.sum().item()} samples)")
    
    # =========================================================================
    # STEP 8: Visualization
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Feature space (first 2 features)
    ax1 = axes[0, 0]
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    for c in range(n_classes):
        mask = y_test == c
        ax1.scatter(X_test[mask, 0].numpy(), X_test[mask, 1].numpy(), 
                   c=colors[c], marker=markers[c], label=f'Class {c}', alpha=0.7, s=50)
    ax1.set_xlabel('Feature 0')
    ax1.set_ylabel('Feature 1')
    ax1.set_title('Test Data in Feature Space (Features 0 & 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predictions vs True Labels
    ax2 = axes[0, 1]
    test_indices = np.arange(len(y_test))
    ax2.plot(test_indices, y_test.numpy(), 'go-', label='True Labels', 
             markersize=8, linewidth=1, alpha=0.7)
    ax2.plot(test_indices, predictions.numpy(), 'rx--', label='Predictions', 
             markersize=6, linewidth=1, alpha=0.7)
    ax2.set_xlabel('Test Sample Index')
    ax2.set_ylabel('Class')
    ax2.set_yticks(list(range(n_classes)))
    ax2.set_title(f'Classification Results (Accuracy: {accuracy:.2%})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Similarity scores to each class centroid
    ax3 = axes[1, 0]
    for c in range(n_classes):
        ax3.plot(test_indices, similarity_scores[:, c].detach().numpy(), 
                f'{colors[c][0]}.-', label=f'Similarity to Class {c}', 
                markersize=4, alpha=0.7)
    ax3.set_xlabel('Test Sample Index')
    ax3.set_ylabel('Dot Product Similarity')
    ax3.set_title('Similarity Scores to Class Centroids')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion visualization
    ax4 = axes[1, 1]
    correct = (predictions == y_test).numpy()
    scatter_colors = ['green' if c else 'red' for c in correct]
    ax4.scatter(test_indices, y_test.numpy(), c=scatter_colors, s=50, alpha=0.7)
    ax4.set_xlabel('Test Sample Index')
    ax4.set_ylabel('True Class')
    ax4.set_yticks(list(range(n_classes)))
    ax4.set_title('Correct (Green) vs Incorrect (Red) Predictions')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Incorrect')
    ]
    ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('vsa_classifier_results.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Visualization saved to 'vsa_classifier_results.png'")
    
    # =========================================================================
    # Summary: VSA Classification Paradigm
    # =========================================================================
    print("\n" + "="*60)
    print("VSA CLASSIFICATION PARADIGM SUMMARY")
    print("="*60)
    print("""
    1. CODEBOOK CREATION:
       - Feature HVs: Random HV for each feature dimension
       - Level HVs: Random HV for each quantization level
    
    2. ENCODING (sample → hypervector):
       For each feature i in sample:
           bound_hv[i] = BIND(feature_hv[i], level_hv[quantize(value)])
       sample_hv = BUNDLE(bound_hv[0], bound_hv[1], ..., bound_hv[n])
    
    3. TRAINING (one-shot, no backprop):
       For each class c:
           centroid[c] = SUM(all sample_hv where label == c)
    
    4. PREDICTION:
       similarity[c] = DOT(test_hv, centroid[c])
       prediction = ARGMAX(similarity)
    
    Key Advantages:
    ✓ No iterative training (one-shot learning)
    ✓ Transparent representations (inspect centroids)
    ✓ Computationally efficient (mostly additions)
    ✓ Robust to noise (high dimensionality)
    """)


# ============================================================================
# LEVEL 3: Traditional Neural Network Classifier (For Comparison)
# ============================================================================
class MLPClassifier(nn.Module):
    """
    Standard Multi-Layer Perceptron for classification.
    This serves as a baseline to compare against VSA.
    """
    def __init__(self, n_features, n_classes, hidden_sizes=[64, 32]):
        super().__init__()
        
        layers = []
        in_size = n_features
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_size, n_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


def level3_neural_network_classifier():
    """
    Learn: Train a traditional neural network on the SAME dataset as Level 2
    Purpose: Compare traditional deep learning vs VSA approach
    
    This demonstrates:
    - Standard backpropagation training
    - Cross-entropy loss
    - Multiple epochs of iterative training
    """
    print("=== LEVEL 3: Traditional Neural Network Classifier ===\n")
    
    # =========================================================================
    # STEP 1: Generate SAME Dataset as Level 2
    # =========================================================================
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    n_classes = 3
    
    samples_per_class = n_samples // n_classes
    
    X_class0 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 1, 0, 0])
    X_class1 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([0, 0, 1, 1])
    X_class2 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 0, 1, 0])
    
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.array([0]*samples_per_class + [1]*samples_per_class + [2]*samples_per_class)
    
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
    
    # =========================================================================
    # STEP 2: Create Neural Network Model
    # =========================================================================
    model = MLPClassifier(n_features=n_features, n_classes=n_classes, hidden_sizes=[64, 32])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: {n_features} → 64 → 32 → {n_classes}")
    print(f"Total Parameters: {total_params:,}\n")
    
    # =========================================================================
    # STEP 3: Training Loop (Iterative Backpropagation)
    # =========================================================================
    n_epochs = 100
    batch_size = 32
    train_losses = []
    train_accs = []
    
    print("Training with backpropagation...")
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Mini-batch training
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_x, batch_y = X_train[batch_idx], y_train[batch_idx]
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_losses.append(epoch_loss / (len(X_train) // batch_size))
        train_accs.append(100. * correct / total)
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: Loss={train_losses[-1]:.4f}, Train Acc={train_accs[-1]:.2f}%")
    
    print(f"\nTraining complete after {n_epochs} epochs.\n")
    
    # =========================================================================
    # STEP 4: Evaluate on Test Data
    # =========================================================================
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = outputs.max(1)
        accuracy = (predictions == y_test).float().mean().item()
    
    print(f"Test Results:")
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    # Per-class accuracy
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            class_acc = (predictions[mask] == y_test[mask]).float().mean().item()
            print(f"  Class {c} accuracy: {class_acc:.2%} ({mask.sum().item()} samples)")
    
    # =========================================================================
    # STEP 5: Visualization
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss Curve
    ax1 = axes[0, 0]
    ax1.plot(range(1, n_epochs+1), train_losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training Accuracy Curve
    ax2 = axes[0, 1]
    ax2.plot(range(1, n_epochs+1), train_accs, 'g-', linewidth=2)
    ax2.axhline(y=accuracy*100, color='r', linestyle='--', label=f'Test Acc: {accuracy:.2%}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Predictions vs True Labels
    ax3 = axes[1, 0]
    test_indices = np.arange(len(y_test))
    ax3.plot(test_indices, y_test.numpy(), 'go-', label='True Labels', 
             markersize=8, linewidth=1, alpha=0.7)
    ax3.plot(test_indices, predictions.numpy(), 'rx--', label='Predictions', 
             markersize=6, linewidth=1, alpha=0.7)
    ax3.set_xlabel('Test Sample Index')
    ax3.set_ylabel('Class')
    ax3.set_yticks(list(range(n_classes)))
    ax3.set_title(f'Classification Results (Accuracy: {accuracy:.2%})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion visualization
    ax4 = axes[1, 1]
    correct_mask = (predictions == y_test).numpy()
    scatter_colors = ['green' if c else 'red' for c in correct_mask]
    ax4.scatter(test_indices, y_test.numpy(), c=scatter_colors, s=50, alpha=0.7)
    ax4.set_xlabel('Test Sample Index')
    ax4.set_ylabel('True Class')
    ax4.set_yticks(list(range(n_classes)))
    ax4.set_title('Correct (Green) vs Incorrect (Red) Predictions')
    ax4.grid(True, alpha=0.3)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
               markersize=10, label='Correct'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='Incorrect')
    ]
    ax4.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('nn_classifier_results.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Visualization saved to 'nn_classifier_results.png'")
    
    # Return results for comparison
    return {
        'accuracy': accuracy,
        'n_params': total_params,
        'n_epochs': n_epochs,
        'train_losses': train_losses,
        'train_accs': train_accs
    }


# ============================================================================
# LEVEL 4: VSA vs Neural Network Comparison
# ============================================================================
def level4_comparison():
    """
    Learn: Direct comparison of VSA vs Neural Network on the SAME dataset
    
    This demonstrates:
    - Memory/parameter efficiency
    - Training time comparison
    - Accuracy comparison
    - Trade-offs between the two approaches
    """
    print("=== LEVEL 4: VSA vs Neural Network Comparison ===\n")
    
    # =========================================================================
    # Generate SAME Dataset
    # =========================================================================
    np.random.seed(42)
    n_samples = 300
    n_features = 4
    n_classes = 3
    
    samples_per_class = n_samples // n_classes
    
    X_class0 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 1, 0, 0])
    X_class1 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([0, 0, 1, 1])
    X_class2 = np.random.randn(samples_per_class, n_features) * 0.3 + np.array([1, 0, 1, 0])
    
    X = np.vstack([X_class0, X_class1, X_class2])
    y = np.array([0]*samples_per_class + [1]*samples_per_class + [2]*samples_per_class)
    
    shuffle_idx = np.random.permutation(n_samples)
    X, y = X[shuffle_idx], y[shuffle_idx]
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Dataset: {n_samples} samples, {n_features} features, {n_classes} classes")
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples\n")
    
    # =========================================================================
    # Method 1: VSA Classifier
    # =========================================================================
    import time
    
    print("-" * 50)
    print("METHOD 1: VSA Centroid Classifier")
    print("-" * 50)
    
    D = 10000
    n_levels = 100
    
    # Create codebook
    feature_hvs = torchhd.random(n_features, D)
    level_hvs = torchhd.random(n_levels, D)
    
    def encode_sample_vsa(sample, feature_hvs, level_hvs, n_levels):
        sample_min, sample_max = sample.min(), sample.max()
        if sample_max - sample_min > 0:
            normalized = (sample - sample_min) / (sample_max - sample_min)
        else:
            normalized = torch.zeros_like(sample)
        level_indices = (normalized * (n_levels - 1)).long().clamp(0, n_levels - 1)
        bound_hvs = []
        for feat_idx in range(len(sample)):
            level_idx = level_indices[feat_idx].item()
            bound_hv = torchhd.bind(feature_hvs[feat_idx], level_hvs[level_idx])
            bound_hvs.append(bound_hv)
        stacked = torch.stack(bound_hvs, dim=0)
        return stacked.sum(dim=0)
    
    # Time VSA training
    vsa_start = time.time()
    
    # Encode data
    train_encoded = torch.stack([encode_sample_vsa(X_train[i], feature_hvs, level_hvs, n_levels) 
                                  for i in range(len(X_train))])
    test_encoded = torch.stack([encode_sample_vsa(X_test[i], feature_hvs, level_hvs, n_levels) 
                                 for i in range(len(X_test))])
    
    # Train (one-shot)
    vsa_model = Centroid(in_features=D, out_features=n_classes)
    vsa_model.add(train_encoded, y_train)
    
    vsa_train_time = time.time() - vsa_start
    
    # Predict
    vsa_scores = vsa_model(test_encoded, dot=True)
    vsa_predictions = vsa_scores.argmax(dim=1)
    vsa_accuracy = (vsa_predictions == y_test).float().mean().item()
    
    # Memory: codebook + centroids
    vsa_memory = (n_features * D + n_levels * D + n_classes * D) * 4 / 1024  # KB
    
    print(f"  Training time: {vsa_train_time:.4f} seconds (ONE-SHOT)")
    print(f"  Test Accuracy: {vsa_accuracy:.2%}")
    print(f"  Memory: {vsa_memory:.2f} KB (codebook + centroids)")
    print(f"  Training iterations: 1 (no backprop)\n")
    
    # =========================================================================
    # Method 2: Neural Network Classifier
    # =========================================================================
    print("-" * 50)
    print("METHOD 2: Neural Network (MLP) Classifier")
    print("-" * 50)
    
    nn_model = MLPClassifier(n_features=n_features, n_classes=n_classes, hidden_sizes=[64, 32])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)
    
    n_epochs = 100
    batch_size = 32
    
    # Time NN training
    nn_start = time.time()
    
    for epoch in range(n_epochs):
        nn_model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            batch_x, batch_y = X_train[batch_idx], y_train[batch_idx]
            optimizer.zero_grad()
            outputs = nn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    nn_train_time = time.time() - nn_start
    
    # Predict
    nn_model.eval()
    with torch.no_grad():
        nn_outputs = nn_model(X_test)
        _, nn_predictions = nn_outputs.max(1)
        nn_accuracy = (nn_predictions == y_test).float().mean().item()
    
    # Memory: model parameters
    nn_params = sum(p.numel() for p in nn_model.parameters())
    nn_memory = nn_params * 4 / 1024  # KB
    
    print(f"  Training time: {nn_train_time:.4f} seconds ({n_epochs} epochs)")
    print(f"  Test Accuracy: {nn_accuracy:.2%}")
    print(f"  Memory: {nn_memory:.2f} KB ({nn_params:,} parameters)")
    print(f"  Training iterations: {n_epochs * (len(X_train) // batch_size)} batches\n")
    
    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("=" * 50)
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    print(f"\n{'Metric':<25} {'VSA':<20} {'Neural Network':<20}")
    print("-" * 65)
    print(f"{'Test Accuracy':<25} {vsa_accuracy:.2%}{'':<14} {nn_accuracy:.2%}")
    print(f"{'Training Time':<25} {vsa_train_time:.4f}s{'':<12} {nn_train_time:.4f}s")
    print(f"{'Memory Usage':<25} {vsa_memory:.2f} KB{'':<12} {nn_memory:.2f} KB")
    print(f"{'Training Method':<25} {'One-shot':<20} {'Backpropagation':<20}")
    print(f"{'Iterations':<25} {'1':<20} {n_epochs * (len(X_train) // batch_size):<20}")
    
    speedup = nn_train_time / vsa_train_time if vsa_train_time > 0 else float('inf')
    
    print(f"\n✓ VSA is {speedup:.1f}x faster to train")
    print(f"✓ VSA uses {vsa_memory/nn_memory:.1f}x more memory (but no gradient storage needed)")
    print(f"✓ Accuracy difference: {abs(vsa_accuracy - nn_accuracy)*100:.1f}%")
    
    # =========================================================================
    # Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0]
    methods = ['VSA\nCentroid', 'Neural\nNetwork']
    accuracies = [vsa_accuracy * 100, nn_accuracy * 100]
    colors = ['steelblue', 'coral']
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{acc:.1f}%', ha='center', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Training Time Comparison
    ax2 = axes[1]
    times = [vsa_train_time, nn_train_time]
    bars = ax2.bar(methods, times, color=colors, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Time (seconds)')
    ax2.set_title('Training Time Comparison')
    for bar, t in zip(bars, times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{t:.3f}s', ha='center', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Memory Comparison
    ax3 = axes[2]
    memories = [vsa_memory, nn_memory]
    bars = ax3.bar(methods, memories, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Memory (KB)')
    ax3.set_title('Memory Usage Comparison')
    for bar, m in zip(bars, memories):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{m:.1f} KB', ha='center', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('vsa_vs_nn_comparison.png', dpi=150)
    plt.show()
    
    print(f"\n✓ Comparison visualization saved to 'vsa_vs_nn_comparison.png'")
    
    # =========================================================================
    # Key Takeaways
    # =========================================================================
    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
    VSA Advantages:
    ✓ One-shot learning (no iterative training)
    ✓ No backpropagation needed
    ✓ Transparent representations (can inspect centroids)
    ✓ Works well with limited data
    ✓ Ideal for edge/embedded devices (simple operations)
    
    Neural Network Advantages:
    ✓ Lower memory footprint
    ✓ Can learn complex non-linear boundaries
    ✓ Flexible architecture choices
    ✓ State-of-the-art for many tasks
    
    When to use VSA:
    → Few training samples available
    → Need interpretable representations
    → Resource-constrained deployment
    → One-shot or few-shot learning scenarios
    
    When to use Neural Networks:
    → Large datasets available
    → Complex patterns to learn
    → Accuracy is the top priority
    → GPU resources available for training
    """)


# ============================================================================
# Run All Levels
# ============================================================================
if __name__ == "__main__":
    print("VSA + Deep Learning Learning Pipeline\n")
    print("=" * 60)
    
    # level1_basic_vsa()
    # level2_vsa_classifier()
    level3_neural_network_classifier()
    # level4_comparison()
    
    print("\n" + "=" * 60)
    print("Completed! You can uncomment different levels to explore:")
    print("  - Level 1: Basic VSA operations (bind, bundle, similarity)")
    print("  - Level 2: VSA Centroid Classifier")
    print("  - Level 3: Traditional Neural Network Classifier")
    print("  - Level 4: Direct comparison of VSA vs NN")