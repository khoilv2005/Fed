"""
STEP 2: Create Non-IID Federated Splits
========================================
Script này chia data thành Non-IID splits cho các clients:
1. Load preprocessed chunks từ Step 1
2. Tạo Non-IID split bằng Dirichlet distribution
3. Save client data với float16
4. Tạo visualization

Memory-efficient: Load chunks theo batch, không load toàn bộ
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
import glob
import logging
import pickle
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'chunks_dir': './data/preprocessed_chunks',
    'output_dir': './data/federated_splits',
    'num_clients': 5,
    'non_iid_alpha': 0.5,
    'test_size': 0.3,
    'create_visualizations': True,
}

# ============================================================================
# LOAD CHUNKS
# ============================================================================

def load_chunk_metadata(chunks_dir):
    """
    Load metadata về chunks (không load data)
    """
    logger.info("="*80)
    logger.info("LOADING CHUNK METADATA")
    logger.info("="*80)

    chunk_files = sorted(glob.glob(os.path.join(chunks_dir, "chunk_*.npz")))

    if len(chunk_files) == 0:
        raise FileNotFoundError(
            f"No chunks found in {chunks_dir}\n"
            f"Please run step1_prepare_chunks.py first"
        )

    logger.info(f"Found {len(chunk_files)} chunks")

    # Load just to get labels for Non-IID split
    all_labels = []
    chunk_info = []

    for i, chunk_file in enumerate(chunk_files):
        data = np.load(chunk_file)
        y_chunk = data['y']

        chunk_info.append({
            'path': chunk_file,
            'num_samples': len(y_chunk),
            'start_idx': len(all_labels),
            'end_idx': len(all_labels) + len(y_chunk)
        })

        all_labels.extend(y_chunk)

    all_labels = np.array(all_labels)

    logger.info(f"✓ Total samples: {len(all_labels):,}")
    logger.info(f"  Number of classes: {len(np.unique(all_labels))}")
    logger.info(f"  Label distribution: {np.bincount(all_labels)}")

    return chunk_info, all_labels

# ============================================================================
# NON-IID SPLIT
# ============================================================================

def create_non_iid_indices(y, num_clients, alpha):
    """
    Tạo Non-IID indices cho các clients
    Chỉ chia indices, không chia data
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING NON-IID SPLIT (INDICES ONLY)")
    logger.info("="*80)
    logger.info(f"Number of clients: {num_clients}")
    logger.info(f"Dirichlet alpha: {alpha}")
    logger.info(f"Total samples: {len(y):,}")

    num_classes = len(np.unique(y))
    logger.info(f"Number of classes: {num_classes}")

    # Dirichlet distribution for Non-IID split
    client_indices = [[] for _ in range(num_clients)]

    for class_id in range(num_classes):
        # Get indices for this class
        class_indices = np.where(y == class_id)[0]
        np.random.shuffle(class_indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        proportions = np.concatenate([[0], proportions])

        # Split indices according to proportions
        for client_id in range(num_clients):
            start = proportions[client_id]
            end = proportions[client_id + 1]
            client_indices[client_id].extend(class_indices[start:end])

    # Shuffle each client's indices
    for client_id in range(num_clients):
        client_indices[client_id] = np.array(client_indices[client_id])
        np.random.shuffle(client_indices[client_id])

        logger.info(f"  Client {client_id}: {len(client_indices[client_id]):,} samples")

    logger.info("\n✓ Non-IID indices created")

    return client_indices

# ============================================================================
# SAVE CLIENT DATA
# ============================================================================

def save_client_data(chunk_info, client_indices, y_all, output_dir, test_size):
    """
    Load chunks và save client data
    """
    logger.info("\n" + "="*80)
    logger.info("SAVING CLIENT DATA")
    logger.info("="*80)

    os.makedirs(output_dir, exist_ok=True)

    num_clients = len(client_indices)
    client_stats = []

    for client_id in range(num_clients):
        logger.info(f"\nClient {client_id}:")

        indices = client_indices[client_id]
        logger.info(f"  Total samples: {len(indices):,}")

        # Collect data for this client from chunks
        X_client = []
        y_client = []

        for chunk in chunk_info:
            # Find indices that belong to this chunk
            chunk_start = chunk['start_idx']
            chunk_end = chunk['end_idx']

            # Get indices in this chunk
            mask = (indices >= chunk_start) & (indices < chunk_end)
            chunk_indices = indices[mask]

            if len(chunk_indices) == 0:
                continue

            # Load chunk
            data = np.load(chunk['path'])
            X_chunk = data['X']
            y_chunk = data['y']

            # Extract samples for this client
            local_indices = chunk_indices - chunk_start
            X_client.append(X_chunk[local_indices])
            y_client.append(y_chunk[local_indices])

        # Concatenate all chunks for this client
        X_client = np.vstack(X_client)
        y_client = np.concatenate(y_client)

        logger.info(f"  Collected data: {X_client.shape}")

        # Split into train/test
        unique, counts = np.unique(y_client, return_counts=True)
        can_stratify = len(unique) > 1 and np.all(counts >= 2)

        X_train, X_test, y_train, y_test = train_test_split(
            X_client, y_client,
            test_size=test_size,
            random_state=42 + client_id,
            stratify=y_client if can_stratify else None
        )

        # Save
        save_path = os.path.join(output_dir, f"client_{client_id}_data.npz")
        np.savez_compressed(
            save_path,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        file_size_mb = os.path.getsize(save_path) / (1024**2)

        logger.info(f"  ✓ Saved to: {save_path}")
        logger.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")

        # Collect stats
        num_classes = len(np.unique(y_all))
        train_dist = np.bincount(y_train, minlength=num_classes)
        test_dist = np.bincount(y_test, minlength=num_classes)

        client_stats.append({
            'client_id': client_id,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X_train) + len(X_test),
            'train_dist': train_dist,
            'test_dist': test_dist,
            'file_size_mb': file_size_mb,
            'memory_mb': (X_train.nbytes + X_test.nbytes) / (1024**2)
        })

        # Clear memory
        del X_client, y_client, X_train, X_test, y_train, y_test
        gc.collect()

    return client_stats

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualizations(client_stats, save_dir):
    """
    Tạo visualization (same as before)
    """
    logger.info("\n" + "="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*80)

    num_clients = len(client_stats)

    fig = plt.figure(figsize=(20, 14))

    # 1. Sample Distribution
    ax1 = plt.subplot(3, 3, 1)
    clients = [f"Client {s['client_id']}" for s in client_stats]
    train_counts = [s['train_samples'] for s in client_stats]
    test_counts = [s['test_samples'] for s in client_stats]

    x = np.arange(len(clients))
    width = 0.35

    bars1 = ax1.bar(x - width/2, train_counts, width, label='Train', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, test_counts, width, label='Test', color='#e74c3c', alpha=0.8)

    ax1.set_xlabel('Client', fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontweight='bold')
    ax1.set_title('1. Sample Distribution per Client', fontweight='bold', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(clients)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom', fontsize=8)

    # 2. Pie Chart
    ax2 = plt.subplot(3, 3, 2)
    total_samples = [s['total_samples'] for s in client_stats]
    colors = plt.cm.Set3(np.linspace(0, 1, num_clients))

    wedges, texts, autotexts = ax2.pie(
        total_samples, labels=clients,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*sum(total_samples)):,})',
        colors=colors, startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax2.set_title('2. Total Sample Distribution', fontweight='bold', fontsize=12)

    # 3. Memory Usage
    ax3 = plt.subplot(3, 3, 3)
    memory_usage = [s['memory_mb'] for s in client_stats]
    file_sizes = [s['file_size_mb'] for s in client_stats]

    bars1 = ax3.bar(x - width/2, memory_usage, width, label='Memory', color='#2ecc71', alpha=0.8)
    bars2 = ax3.bar(x + width/2, file_sizes, width, label='File Size', color='#9b59b6', alpha=0.8)

    ax3.set_xlabel('Client', fontweight='bold')
    ax3.set_ylabel('Size (MB)', fontweight='bold')
    ax3.set_title('3. Memory Usage & File Size', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(clients)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # 4-9. Class Distribution
    for idx, client_stat in enumerate(client_stats):
        ax = plt.subplot(3, 3, 4 + idx) if idx < 3 else plt.subplot(3, 3, 7 + (idx - 3))

        train_dist = client_stat['train_dist']
        test_dist = client_stat['test_dist']

        classes_with_data = np.where((train_dist + test_dist) > 0)[0]

        if len(classes_with_data) > 15:
            total_dist = train_dist + test_dist
            top_classes = np.argsort(total_dist)[-15:][::-1]
            classes_to_show = top_classes
        else:
            classes_to_show = classes_with_data

        x_pos = np.arange(len(classes_to_show))

        ax.bar(x_pos - width/2, train_dist[classes_to_show], width,
               label='Train', color='#3498db', alpha=0.8)
        ax.bar(x_pos + width/2, test_dist[classes_to_show], width,
               label='Test', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Class ID', fontweight='bold')
        ax.set_ylabel('Samples', fontweight='bold')
        ax.set_title(f'Client {idx} Class Distribution', fontweight='bold', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(c) for c in classes_to_show], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Federated Learning - Non-IID Data Distribution',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    vis_path = os.path.join(save_dir, "federated_data_distribution.png")
    plt.savefig(vis_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Visualization saved: {vis_path}")
    plt.close()

    # Create heatmap
    create_heatmap(client_stats, save_dir)

def create_heatmap(client_stats, save_dir):
    """
    Tạo heatmap
    """
    num_clients = len(client_stats)
    num_classes = len(client_stats[0]['train_dist'])

    distribution_matrix = np.zeros((num_clients, num_classes))

    for i, client_stat in enumerate(client_stats):
        total_dist = client_stat['train_dist'] + client_stat['test_dist']
        distribution_matrix[i] = total_dist

    classes_with_data = np.where(distribution_matrix.sum(axis=0) > 0)[0]
    distribution_matrix = distribution_matrix[:, classes_with_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    sns.heatmap(np.log10(distribution_matrix + 1), ax=ax1, cmap='YlOrRd',
                xticklabels=[f'C{c}' for c in classes_with_data],
                yticklabels=[f'Client {i}' for i in range(num_clients)],
                cbar_kws={'label': 'log10(Count + 1)'}, linewidths=0.5)
    ax1.set_title('Class Distribution (Log Scale)', fontweight='bold', fontsize=14)

    row_sums = distribution_matrix.sum(axis=1, keepdims=True)
    distribution_pct = (distribution_matrix / row_sums) * 100

    sns.heatmap(distribution_pct, ax=ax2, cmap='Blues',
                xticklabels=[f'C{c}' for c in classes_with_data],
                yticklabels=[f'Client {i}' for i in range(num_clients)],
                cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5)
    ax2.set_title('Class Distribution (%)', fontweight='bold', fontsize=14)

    plt.suptitle('Non-IID Class Distribution Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()

    heatmap_path = os.path.join(save_dir, "class_distribution_heatmap.png")
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    logger.info(f"✓ Heatmap saved: {heatmap_path}")
    plt.close()

# ============================================================================
# MAIN
# ============================================================================

def main():
    logger.info("="*80)
    logger.info("STEP 2: CREATE NON-IID FEDERATED SPLITS")
    logger.info("="*80)
    logger.info("\nThis script will:")
    logger.info("  1. Load preprocessed chunks")
    logger.info("  2. Create Non-IID split indices")
    logger.info("  3. Save client data (float16)")
    logger.info("  4. Create visualizations")
    logger.info("\nMemory requirement: ~2-4 GB RAM")
    logger.info("="*80)

    config = CONFIG

    logger.info(f"\nConfiguration:")
    logger.info(f"  Chunks directory: {config['chunks_dir']}")
    logger.info(f"  Output directory: {config['output_dir']}")
    logger.info(f"  Number of clients: {config['num_clients']}")
    logger.info(f"  Non-IID alpha: {config['non_iid_alpha']}")
    logger.info(f"  Test size: {config['test_size']}")

    # Step 1: Load chunk metadata
    chunk_info, y_all = load_chunk_metadata(config['chunks_dir'])

    # Step 2: Create Non-IID indices
    client_indices = create_non_iid_indices(
        y=y_all,
        num_clients=config['num_clients'],
        alpha=config['non_iid_alpha']
    )

    # Step 3: Save client data
    client_stats = save_client_data(
        chunk_info=chunk_info,
        client_indices=client_indices,
        y_all=y_all,
        output_dir=config['output_dir'],
        test_size=config['test_size']
    )

    # Step 4: Create visualizations
    if config['create_visualizations']:
        create_visualizations(client_stats, config['output_dir'])

    logger.info("\n" + "="*80)
    logger.info("✓ STEP 2 COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutput:")
    logger.info(f"  Directory: {config['output_dir']}")
    logger.info(f"  Files: client_0_data.npz to client_{config['num_clients']-1}_data.npz")
    logger.info(f"  Visualizations: federated_data_distribution.png, class_distribution_heatmap.png")
    logger.info("\nNext step:")
    logger.info("  python test_presplit_data.py")
    logger.info("  or start training with pre-split data")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
