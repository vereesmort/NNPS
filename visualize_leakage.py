#!/usr/bin/env python
# coding: utf-8
"""
Visualization script to demonstrate data leakage in NNPS_check_leakage.py
Shows that duplicate pairs ([d1,d2] and [d2,d1]) can end up in different splits
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

def normalize_pair(pair):
    """Normalize pair to always have smaller index first"""
    return tuple(sorted(pair))

def find_leakage(train, val, test):
    """Find pairs that appear in multiple splits"""
    train_normalized = {normalize_pair(p) for p in train}
    val_normalized = {normalize_pair(p) for p in val}
    test_normalized = {normalize_pair(p) for p in test}
    
    train_val_leak = train_normalized & val_normalized
    train_test_leak = train_normalized & test_normalized
    val_test_leak = val_normalized & test_normalized
    all_leak = train_normalized & val_normalized & test_normalized
    
    return {
        'train_val': train_val_leak,
        'train_test': train_test_leak,
        'val_test': val_test_leak,
        'all_splits': all_leak
    }

def visualize_data_leakage(n_drugs=50, n_edges_per_se=20, val_test_size=0.05, seed=42):
    """
    Simulate the data splitting process and visualize leakage
    
    Parameters:
    -----------
    n_drugs : int
        Number of drugs to simulate
    n_edges_per_se : int
        Number of edges per side effect (will be doubled due to duplicates)
    val_test_size : float
        Fraction of data for validation/test
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print("=" * 80)
    print("DATA LEAKAGE VISUALIZATION")
    print("=" * 80)
    print(f"\nSimulating with {n_drugs} drugs, {n_edges_per_se} unique edges per side effect")
    
    # Simulate one side effect's adjacency matrix
    # Create symmetric adjacency matrix with some edges
    adj_matrix = np.zeros((n_drugs, n_drugs))
    
    # Add random edges (symmetric)
    edge_count = 0
    while edge_count < n_edges_per_se:
        i, j = np.random.randint(0, n_drugs, 2)
        if i != j and adj_matrix[i, j] == 0:
            adj_matrix[i, j] = adj_matrix[j, i] = 1
            edge_count += 1
    
    # Extract edges as done in the original code (lines 184-187)
    # This creates duplicates: each edge appears as both [i,j] and [j,i]
    edges = []
    for i in range(n_drugs):
        for j in range(n_drugs):
            if adj_matrix[i, j] == 1:
                edges.append([i, j])
    
    print(f"\n1. EDGE EXTRACTION:")
    print(f"   Total edges extracted: {len(edges)}")
    print(f"   Expected unique edges: {n_edges_per_se}")
    print(f"   Duplicate factor: {len(edges) / n_edges_per_se:.1f}x")
    
    # Check for duplicate pairs
    pair_counts = defaultdict(list)
    for idx, edge in enumerate(edges):
        normalized = normalize_pair(edge)
        pair_counts[normalized].append((idx, edge))
    
    duplicates = {k: v for k, v in pair_counts.items() if len(v) > 1}
    
    print(f"\n2. DUPLICATE PAIRS DETECTED:")
    print(f"   Unique pairs (normalized): {len(pair_counts)}")
    print(f"   Duplicate pairs found: {len(duplicates)}")
    print(f"   Percentage of pairs that are duplicated: {len(duplicates)/len(pair_counts)*100:.1f}%")
    
    if duplicates:
        print(f"\n   Examples of duplicate pairs:")
        for i, (norm_pair, occurrences) in enumerate(list(duplicates.items())[:5]):
            print(f"   Pair {norm_pair}: appears as {occurrences}")
    
    # Simulate the splitting process (as in original code lines 197-218)
    np.random.shuffle(edges)
    
    # Create negative edges (simplified - just use some non-edges)
    edges_false = []
    false_count = 0
    while false_count < len(edges):
        i, j = np.random.randint(0, n_drugs, 2)
        if i != j and adj_matrix[i, j] == 0:
            edges_false.append([i, j])
            false_count += 1
    
    np.random.shuffle(edges_false)
    edges_false = edges_false[:len(edges)]
    
    edges_all = edges + edges_false
    np.random.shuffle(edges_all)
    edges_all = edges_all[:len(edges)]  # Keep same size as positive edges
    
    # Split into train/val/test
    split_point = int(np.floor(len(edges_all) * val_test_size))
    val = edges_all[:split_point]
    test = edges_all[split_point:split_point*2]
    train = edges_all[split_point*2:]
    
    print(f"\n3. SPLIT DISTRIBUTION:")
    print(f"   Train: {len(train)} edges")
    print(f"   Val: {len(val)} edges")
    print(f"   Test: {len(test)} edges")
    
    # Check for leakage
    leakage = find_leakage(train, val, test)
    
    print(f"\n4. DATA LEAKAGE DETECTED:")
    print(f"   Pairs in both Train & Val: {len(leakage['train_val'])}")
    print(f"   Pairs in both Train & Test: {len(leakage['train_test'])}")
    print(f"   Pairs in both Val & Test: {len(leakage['val_test'])}")
    print(f"   Pairs in all three splits: {len(leakage['all_splits'])}")
    
    if leakage['train_test']:
        print(f"\n   Example leaked pairs (Train & Test):")
        for pair in list(leakage['train_test'])[:5]:
            train_repr = [p for p in train if normalize_pair(p) == pair]
            test_repr = [p for p in test if normalize_pair(p) == pair]
            print(f"   Pair {pair}:")
            print(f"     In train as: {train_repr}")
            print(f"     In test as: {test_repr}")
    
    # Create visualizations
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Duplicate pairs histogram
    ax1 = plt.subplot(2, 3, 1)
    duplicate_counts = [len(v) for v in duplicates.values()]
    if duplicate_counts:
        ax1.hist(duplicate_counts, bins=range(1, max(duplicate_counts)+2), 
                edgecolor='black', alpha=0.7, color='coral')
        ax1.set_xlabel('Number of Occurrences')
        ax1.set_ylabel('Number of Pairs')
        ax1.set_title('Distribution of Duplicate Occurrences\n(Same pair in different orders)')
        ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Leakage across splits
    ax2 = plt.subplot(2, 3, 2)
    leakage_types = ['Train & Val', 'Train & Test', 'Val & Test', 'All Splits']
    leakage_counts = [
        len(leakage['train_val']),
        len(leakage['train_test']),
        len(leakage['val_test']),
        len(leakage['all_splits'])
    ]
    colors = ['red' if c > 0 else 'green' for c in leakage_counts]
    bars = ax2.bar(leakage_types, leakage_counts, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Number of Leaked Pairs')
    ax2.set_title('Data Leakage Across Splits')
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, leakage_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Split sizes
    ax3 = plt.subplot(2, 3, 3)
    split_names = ['Train', 'Val', 'Test']
    split_sizes = [len(train), len(val), len(test)]
    ax3.bar(split_names, split_sizes, color=['blue', 'orange', 'green'], 
           alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Number of Edges')
    ax3.set_title('Split Sizes')
    ax3.grid(axis='y', alpha=0.3)
    for i, (name, size) in enumerate(zip(split_names, split_sizes)):
        ax3.text(i, size, f'{size}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Pair distribution across splits (Venn-like visualization)
    ax4 = plt.subplot(2, 3, 4)
    train_only = len({normalize_pair(p) for p in train} - 
                     {normalize_pair(p) for p in val} - 
                     {normalize_pair(p) for p in test})
    val_only = len({normalize_pair(p) for p in val} - 
                   {normalize_pair(p) for p in train} - 
                   {normalize_pair(p) for p in test})
    test_only = len({normalize_pair(p) for p in test} - 
                    {normalize_pair(p) for p in train} - 
                    {normalize_pair(p) for p in val})
    train_val = len(leakage['train_val'] - leakage['all_splits'])
    train_test = len(leakage['train_test'] - leakage['all_splits'])
    val_test = len(leakage['val_test'] - leakage['all_splits'])
    all_three = len(leakage['all_splits'])
    
    categories = ['Train\nOnly', 'Val\nOnly', 'Test\nOnly', 
                 'Train&\nVal', 'Train&\nTest', 'Val&\nTest', 'All\nThree']
    counts = [train_only, val_only, test_only, train_val, train_test, val_test, all_three]
    colors_v = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'red']
    bars = ax4.bar(categories, counts, color=colors_v, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Number of Unique Pairs')
    ax4.set_title('Pair Distribution Across Splits\n(Normalized pairs)')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars, counts):
        if count > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Feature vector demonstration
    ax5 = plt.subplot(2, 3, 5)
    # Simulate feature vectors
    np.random.seed(42)
    drug_feat_demo = np.random.rand(10, 5)
    d1, d2 = 0, 1
    feat_01 = drug_feat_demo[d1] + drug_feat_demo[d2]
    feat_10 = drug_feat_demo[d2] + drug_feat_demo[d1]
    
    x = np.arange(len(feat_01))
    width = 0.35
    ax5.bar(x - width/2, feat_01, width, label='[0,1]', alpha=0.7, color='blue')
    ax5.bar(x + width/2, feat_10, width, label='[1,0]', alpha=0.7, color='green')
    ax5.set_xlabel('Feature Dimension')
    ax5.set_ylabel('Feature Value')
    ax5.set_title('Feature Vectors: [0,1] vs [1,0]\n(IDENTICAL due to commutative addition)')
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    stats_text = f"""
    SUMMARY STATISTICS
    
    Total Edges Extracted: {len(edges)}
    Unique Pairs: {len(pair_counts)}
    Duplicate Pairs: {len(duplicates)}
    Duplication Rate: {len(duplicates)/len(pair_counts)*100:.1f}%
    
    LEAKAGE DETECTED:
    Train ↔ Val: {len(leakage['train_val'])} pairs
    Train ↔ Test: {len(leakage['train_test'])} pairs
    Val ↔ Test: {len(leakage['val_test'])} pairs
    All Splits: {len(leakage['all_splits'])} pairs
    
    TOTAL LEAKED PAIRS: {len(leakage['train_val'] | leakage['train_test'] | leakage['val_test'])}
    """
    ax6.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('data_leakage_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\n5. Visualization saved to 'data_leakage_visualization.png'")
    
    plt.show()
    
    return {
        'edges': edges,
        'duplicates': duplicates,
        'leakage': leakage,
        'train': train,
        'val': val,
        'test': test
    }

if __name__ == "__main__":
    # Run visualization
    results = visualize_data_leakage(n_drugs=50, n_edges_per_se=30, val_test_size=0.05, seed=42)
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("=" * 80)
    print("The code has data leakage because:")
    print("  1. Same pairs appear TWICE in the edge list (as [d1, d2] and [d2, d1])")
    print("  2. After shuffling, one copy can end up in TRAIN and the other in TEST/VAL")
    print("  3. Even though feature vectors are identical (commutative addition),")
    print("     having the same example in both splits violates train/test separation")
    print("  4. This artificially inflates the dataset size and causes data leakage")
    print("=" * 80)
