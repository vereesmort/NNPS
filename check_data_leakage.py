#!/usr/bin/env python
# coding: utf-8
"""
Data Leakage Detection Script for NNPS.py
Uses the exact same data splitting code to detect and visualize leakage
"""

from __future__ import division
from __future__ import print_function
from itertools import combinations
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#----------------------------------------------------------------------------
# Data loading functions (from NNPS.py)
def load_combo_se(fname='Datasets/bio-decagon-combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print('Drug-drug interactions: %d' % (n_interactions))
    return combo2stitch, combo2se, se2name

def load_mono_se(fname='Datasets/bio-decagon-mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

def load_targets(fname='Datasets/bio-decagon-targets-all.csv'):
    stitch2proteins_all = defaultdict(set)
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins_all[stitch_id].add(gene)
    return stitch2proteins_all

def get_se_counter(se_map):
    side_effects = []
    for drug in se_map:
        side_effects += list(set(se_map[drug]))
    return Counter(side_effects)

def normalize_pair(pair):
    """Normalize pair to always have smaller index first"""
    return tuple(sorted(pair))

def pair_to_drug_names(pair, drug_list):
    """Convert pair indices to actual drug names"""
    return (drug_list[pair[0]], drug_list[pair[1]])

#----------------------------------------------------------------------------
# Load data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
combo2stitch, combo2se, se2name = load_combo_se()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins_all = load_targets()

# Most common side effects in drug combinations
combo_counter = get_se_counter(combo2se)
print("\nMost common side effects in drug combinations:")
common_se = []
common_se_counts = []
common_se_names = []
for se, count in combo_counter.most_common(964):
    common_se += [se]
    common_se_counts += [count]
    common_se_names += [se2name[se]]
df = pd.DataFrame(data={"Side Effect": common_se, "Frequency in Drug Combos": common_se_counts, "Name": common_se_names})

# Parameters
val_test_size = 0.05
n_drugs = 645
n_proteins = 8934
n_drugdrug_rel_types = 30

# List of drugs
lst = []
for key, value in combo2stitch.items():
    first_name, second_name = map(lambda x: x.strip(), key.split('_'))
    if first_name not in lst:
        lst.append(first_name)
    if second_name not in lst:
        lst.append(second_name)

# List of proteins
p = []
for k, v in stitch2proteins_all.items():
    for i in v:
        if i not in p:
            p.append(i)

# Construct drug-protein-adj matrix
drug_protein_adj = np.zeros((n_drugs, n_proteins))
for i in range(n_drugs):
    for j in stitch2proteins_all[lst[i]]:
        k = p.index(j)
        drug_protein_adj[i, k] = 1

# Construct drug-drug-adj matrices for all side effects
print("\n" + "=" * 80)
print("CONSTRUCTING ADJACENCY MATRICES")
print("=" * 80)
drug_drug_adj_list = []
l = []
for i in range(n_drugdrug_rel_types):
    print(f"Processing side effect {i+1}/{n_drugdrug_rel_types}")
    mat = np.zeros((n_drugs, n_drugs))
    l.append(df.at[i, 'Side Effect'])
    for se in l:
        for d1, d2 in combinations(list(range(n_drugs)), 2):
            if lst[d1] + "_" + lst[d2] in combo2se:
                if se in combo2se[lst[d1] + "_" + lst[d2]]:
                    mat[d1, d2] = mat[d2, d1] = 1
    l = []
    drug_drug_adj_list.append(mat)

#-------------------------------------------------------------------------
# EXACT DATA SPLITTING CODE FROM NNPS.py
print("\n" + "=" * 80)
print("EXTRACTING EDGES AND SPLITTING DATA (EXACT CODE FROM NNPS.py)")
print("=" * 80)
edges = []
for k in range(n_drugdrug_rel_types):
    l = []
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i, j] == 1:
                l.append([i, j])
    edges.append(l)

edges_false = []
for k in range(n_drugdrug_rel_types):
    l = []
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i, j] == 0:
                l.append([i, j])
    edges_false.append(l)

for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges[k])
    np.random.shuffle(edges_false[k])

for k in range(n_drugdrug_rel_types):
    a = len(edges[k])
    edges_false[k] = edges_false[k][:a]

edges_all = []
for k in range(n_drugdrug_rel_types):
    edges_all.append(edges[k] + edges_false[k])

for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges_all[k])

for k in range(n_drugdrug_rel_types):
    a = len(edges[k])
    edges_all[k] = edges_all[k][:a]

val = []
test = []
train = []
for k in range(n_drugdrug_rel_types):
    a = int(np.floor(len(edges_all[k]) * val_test_size))
    val.append(edges_all[k][:a])
    test.append(edges_all[k][a:a+a])
    train.append(edges_all[k][a+a:])

#-------------------------------------------------------------------------
# DETECT DATA LEAKAGE
print("\n" + "=" * 80)
print("DETECTING DATA LEAKAGE")
print("=" * 80)

def find_leakage_detailed(train, val, test, drug_list):
    """Find pairs that appear in multiple splits with detailed information"""
    train_normalized = {normalize_pair(p): p for p in train}
    val_normalized = {normalize_pair(p): p for p in val}
    test_normalized = {normalize_pair(p): p for p in test}
    
    train_val_leak = {}
    train_test_leak = {}
    val_test_leak = {}
    all_leak = {}
    
    for norm_pair in train_normalized:
        if norm_pair in val_normalized:
            train_val_leak[norm_pair] = {
                'train': train_normalized[norm_pair],
                'val': val_normalized[norm_pair],
                'drug_names': pair_to_drug_names(norm_pair, drug_list)
            }
        if norm_pair in test_normalized:
            train_test_leak[norm_pair] = {
                'train': train_normalized[norm_pair],
                'test': test_normalized[norm_pair],
                'drug_names': pair_to_drug_names(norm_pair, drug_list)
            }
        if norm_pair in val_normalized and norm_pair in test_normalized:
            all_leak[norm_pair] = {
                'train': train_normalized[norm_pair],
                'val': val_normalized[norm_pair],
                'test': test_normalized[norm_pair],
                'drug_names': pair_to_drug_names(norm_pair, drug_list)
            }
    
    for norm_pair in val_normalized:
        if norm_pair in test_normalized and norm_pair not in all_leak:
            val_test_leak[norm_pair] = {
                'val': val_normalized[norm_pair],
                'test': test_normalized[norm_pair],
                'drug_names': pair_to_drug_names(norm_pair, drug_list)
            }
    
    return {
        'train_val': train_val_leak,
        'train_test': train_test_leak,
        'val_test': val_test_leak,
        'all_splits': all_leak
    }

# Analyze all side effects
all_leakage_results = []
for k in range(n_drugdrug_rel_types):
    leakage = find_leakage_detailed(train[k], val[k], test[k], lst)
    
    # Count duplicates in edges
    pair_counts = defaultdict(list)
    for idx, edge in enumerate(edges[k]):
        normalized = normalize_pair(edge)
        pair_counts[normalized].append(edge)
    
    duplicates = {k: v for k, v in pair_counts.items() if len(v) > 1}
    unique_pairs = len(pair_counts)
    
    all_leakage_results.append({
        'side_effect_idx': k,
        'side_effect_id': df.at[k, 'Side Effect'],
        'side_effect_name': df.at[k, 'Name'],
        'total_edges': len(edges[k]),
        'unique_pairs': unique_pairs,
        'duplicate_pairs': len(duplicates),
        'duplication_rate': len(duplicates) / unique_pairs * 100 if unique_pairs > 0 else 0,
        'leakage': leakage,
        'train_size': len(train[k]),
        'val_size': len(val[k]),
        'test_size': len(test[k])
    })

# Print summary
print("\nSUMMARY OF DATA LEAKAGE ACROSS ALL SIDE EFFECTS:")
print("-" * 80)
for result in all_leakage_results:
    k = result['side_effect_idx']
    leakage = result['leakage']
    print(f"\nSide Effect {k+1}: {result['side_effect_name']} (ID: {result['side_effect_id']})")
    print(f"  Total edges extracted: {result['total_edges']}")
    print(f"  Unique pairs: {result['unique_pairs']}")
    print(f"  Duplicate pairs: {result['duplicate_pairs']} ({result['duplication_rate']:.1f}%)")
    print(f"  Leakage:")
    print(f"    Train ↔ Val: {len(leakage['train_val'])} pairs")
    print(f"    Train ↔ Test: {len(leakage['train_test'])} pairs")
    print(f"    Val ↔ Test: {len(leakage['val_test'])} pairs")
    print(f"    All three splits: {len(leakage['all_splits'])} pairs")

# Show detailed examples for first side effect
print("\n" + "=" * 80)
print("DETAILED EXAMPLES OF LEAKED PAIRS (Side Effect 1)")
print("=" * 80)
first_result = all_leakage_results[0]
leakage = first_result['leakage']

if leakage['train_test']:
    print(f"\nExamples of pairs that appear in BOTH TRAIN and TEST:")
    print("-" * 80)
    for i, (norm_pair, info) in enumerate(list(leakage['train_test'].items())[:10]):
        print(f"\n{i+1}. Drug Pair: {info['drug_names'][0]} <-> {info['drug_names'][1]}")
        print(f"   Normalized pair: {norm_pair}")
        print(f"   In TRAIN as: {info['train']}")
        print(f"   In TEST as: {info['test']}")

if leakage['train_val']:
    print(f"\n\nExamples of pairs that appear in BOTH TRAIN and VAL:")
    print("-" * 80)
    for i, (norm_pair, info) in enumerate(list(leakage['train_val'].items())[:10]):
        print(f"\n{i+1}. Drug Pair: {info['drug_names'][0]} <-> {info['drug_names'][1]}")
        print(f"   Normalized pair: {norm_pair}")
        print(f"   In TRAIN as: {info['train']}")
        print(f"   In VAL as: {info['val']}")

if leakage['all_splits']:
    print(f"\n\nExamples of pairs that appear in ALL THREE SPLITS:")
    print("-" * 80)
    for i, (norm_pair, info) in enumerate(list(leakage['all_splits'].items())[:10]):
        print(f"\n{i+1}. Drug Pair: {info['drug_names'][0]} <-> {info['drug_names'][1]}")
        print(f"   Normalized pair: {norm_pair}")
        print(f"   In TRAIN as: {info['train']}")
        print(f"   In VAL as: {info['val']}")
        print(f"   In TEST as: {info['test']}")

#-------------------------------------------------------------------------
# CREATE VISUALIZATIONS
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Create output directory
os.makedirs('Leakage_Analysis', exist_ok=True)

# Visualization 1: Summary across all side effects
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Duplication rate across side effects
ax1 = axes[0, 0]
dup_rates = [r['duplication_rate'] for r in all_leakage_results]
ax1.bar(range(n_drugdrug_rel_types), dup_rates, color='coral', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Side Effect Index')
ax1.set_ylabel('Duplication Rate (%)')
ax1.set_title('Duplication Rate Across All Side Effects\n(Same pair appears multiple times)')
ax1.grid(axis='y', alpha=0.3)
ax1.set_xticks(range(0, n_drugdrug_rel_types, 5))

# Plot 2: Leakage counts across side effects
ax2 = axes[0, 1]
train_test_counts = [len(r['leakage']['train_test']) for r in all_leakage_results]
train_val_counts = [len(r['leakage']['train_val']) for r in all_leakage_results]
val_test_counts = [len(r['leakage']['val_test']) for r in all_leakage_results]
all_counts = [len(r['leakage']['all_splits']) for r in all_leakage_results]

x = np.arange(n_drugdrug_rel_types)
width = 0.2
ax2.bar(x - 1.5*width, train_test_counts, width, label='Train ↔ Test', alpha=0.7, color='red')
ax2.bar(x - 0.5*width, train_val_counts, width, label='Train ↔ Val', alpha=0.7, color='orange')
ax2.bar(x + 0.5*width, val_test_counts, width, label='Val ↔ Test', alpha=0.7, color='yellow')
ax2.bar(x + 1.5*width, all_counts, width, label='All Three', alpha=0.7, color='purple')
ax2.set_xlabel('Side Effect Index')
ax2.set_ylabel('Number of Leaked Pairs')
ax2.set_title('Data Leakage Across All Side Effects')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(range(0, n_drugdrug_rel_types, 5))

# Plot 3: Total leaked pairs per side effect
ax3 = axes[1, 0]
total_leaked = []
for r in all_leakage_results:
    total = (len(r['leakage']['train_val']) + 
             len(r['leakage']['train_test']) + 
             len(r['leakage']['val_test']) + 
             len(r['leakage']['all_splits']))
    total_leaked.append(total)
ax3.bar(range(n_drugdrug_rel_types), total_leaked, color='darkred', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Side Effect Index')
ax3.set_ylabel('Total Leaked Pairs')
ax3.set_title('Total Leaked Pairs Per Side Effect')
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(range(0, n_drugdrug_rel_types, 5))

# Plot 4: Edge counts vs unique pairs
ax4 = axes[1, 1]
total_edges = [r['total_edges'] for r in all_leakage_results]
unique_pairs = [r['unique_pairs'] for r in all_leakage_results]
x_pos = np.arange(n_drugdrug_rel_types)
width = 0.35
ax4.bar(x_pos - width/2, total_edges, width, label='Total Edges', alpha=0.7, color='blue')
ax4.bar(x_pos + width/2, unique_pairs, width, label='Unique Pairs', alpha=0.7, color='green')
ax4.set_xlabel('Side Effect Index')
ax4.set_ylabel('Count')
ax4.set_title('Total Edges vs Unique Pairs\n(Shows duplication)')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_xticks(range(0, n_drugdrug_rel_types, 5))

plt.tight_layout()
plt.savefig('Leakage_Analysis/leakage_summary_all_side_effects.png', dpi=300, bbox_inches='tight')
print("Saved: Leakage_Analysis/leakage_summary_all_side_effects.png")

# Visualization 2: Detailed view for first side effect
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

first_result = all_leakage_results[0]
leakage = first_result['leakage']

# Plot 1: Duplicate pairs histogram
ax1 = axes[0, 0]
pair_counts = defaultdict(list)
for idx, edge in enumerate(edges[0]):
    normalized = normalize_pair(edge)
    pair_counts[normalized].append(edge)
duplicate_counts = [len(v) for v in pair_counts.values() if len(v) > 1]
if duplicate_counts:
    ax1.hist(duplicate_counts, bins=range(1, max(duplicate_counts)+2), 
            edgecolor='black', alpha=0.7, color='coral')
    ax1.set_xlabel('Number of Occurrences')
    ax1.set_ylabel('Number of Pairs')
    ax1.set_title('Distribution of Duplicate Occurrences\n(Side Effect 1)')
    ax1.grid(axis='y', alpha=0.3)

# Plot 2: Leakage types
ax2 = axes[0, 1]
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
ax2.set_title('Data Leakage Types\n(Side Effect 1)')
ax2.grid(axis='y', alpha=0.3)
ax2.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, leakage_counts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}', ha='center', va='bottom', fontweight='bold')

# Plot 3: Split sizes
ax3 = axes[0, 2]
split_names = ['Train', 'Val', 'Test']
split_sizes = [first_result['train_size'], first_result['val_size'], first_result['test_size']]
ax3.bar(split_names, split_sizes, color=['blue', 'orange', 'green'], 
       alpha=0.7, edgecolor='black')
ax3.set_ylabel('Number of Edges')
ax3.set_title('Split Sizes\n(Side Effect 1)')
ax3.grid(axis='y', alpha=0.3)
for i, (name, size) in enumerate(zip(split_names, split_sizes)):
    ax3.text(i, size, f'{size}', ha='center', va='bottom', fontweight='bold')

# Plot 4: Pair distribution across splits
ax4 = axes[1, 0]
train_normalized = {normalize_pair(p) for p in train[0]}
val_normalized = {normalize_pair(p) for p in val[0]}
test_normalized = {normalize_pair(p) for p in test[0]}

train_only = len(train_normalized - val_normalized - test_normalized)
val_only = len(val_normalized - train_normalized - test_normalized)
test_only = len(test_normalized - train_normalized - val_normalized)
train_val = len(leakage['train_val']) - len(leakage['all_splits'])
train_test = len(leakage['train_test']) - len(leakage['all_splits'])
val_test = len(leakage['val_test']) - len(leakage['all_splits'])
all_three = len(leakage['all_splits'])

categories = ['Train\nOnly', 'Val\nOnly', 'Test\nOnly', 
             'Train&\nVal', 'Train&\nTest', 'Val&\nTest', 'All\nThree']
counts = [train_only, val_only, test_only, train_val, train_test, val_test, all_three]
colors_v = ['blue', 'orange', 'green', 'purple', 'brown', 'pink', 'red']
bars = ax4.bar(categories, counts, color=colors_v, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Number of Unique Pairs')
ax4.set_title('Pair Distribution Across Splits\n(Side Effect 1)')
ax4.grid(axis='y', alpha=0.3)
ax4.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, counts):
    if count > 0:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom', fontsize=8)

# Plot 5: Statistics summary
ax5 = axes[1, 1]
ax5.axis('off')
stats_text = f"""
SIDE EFFECT 1 STATISTICS

Side Effect: {first_result['side_effect_name']}
ID: {first_result['side_effect_id']}

Total Edges Extracted: {first_result['total_edges']}
Unique Pairs: {first_result['unique_pairs']}
Duplicate Pairs: {first_result['duplicate_pairs']}
Duplication Rate: {first_result['duplication_rate']:.1f}%

LEAKAGE DETECTED:
Train ↔ Val: {len(leakage['train_val'])} pairs
Train ↔ Test: {len(leakage['train_test'])} pairs
Val ↔ Test: {len(leakage['val_test'])} pairs
All Splits: {len(leakage['all_splits'])} pairs

TOTAL LEAKED PAIRS: {len(leakage['train_val']) + len(leakage['train_test']) + len(leakage['val_test']) + len(leakage['all_splits'])}
"""
ax5.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 6: Leakage percentage
ax6 = axes[1, 2]
total_unique_in_splits = len(train_normalized | val_normalized | test_normalized)
total_leaked_unique = len(leakage['train_val']) + len(leakage['train_test']) + len(leakage['val_test']) + len(leakage['all_splits'])
leakage_pct = (total_leaked_unique / total_unique_in_splits * 100) if total_unique_in_splits > 0 else 0

categories_pie = ['Leaked\nPairs', 'Non-Leaked\nPairs']
sizes = [total_leaked_unique, total_unique_in_splits - total_leaked_unique]
colors_pie = ['red', 'green']
ax6.pie(sizes, labels=categories_pie, colors=colors_pie, autopct='%1.1f%%', startangle=90)
ax6.set_title(f'Leakage Percentage\n(Side Effect 1)\n{leakage_pct:.1f}% leaked')

plt.tight_layout()
plt.savefig('Leakage_Analysis/leakage_detailed_side_effect_1.png', dpi=300, bbox_inches='tight')
print("Saved: Leakage_Analysis/leakage_detailed_side_effect_1.png")

# Save detailed leakage information to CSV
print("\n" + "=" * 80)
print("SAVING DETAILED RESULTS TO CSV")
print("=" * 80)

leakage_summary = []
for result in all_leakage_results:
    leakage = result['leakage']
    leakage_summary.append({
        'side_effect_index': result['side_effect_idx'],
        'side_effect_id': result['side_effect_id'],
        'side_effect_name': result['side_effect_name'],
        'total_edges': result['total_edges'],
        'unique_pairs': result['unique_pairs'],
        'duplicate_pairs': result['duplicate_pairs'],
        'duplication_rate': result['duplication_rate'],
        'train_val_leakage': len(leakage['train_val']),
        'train_test_leakage': len(leakage['train_test']),
        'val_test_leakage': len(leakage['val_test']),
        'all_splits_leakage': len(leakage['all_splits']),
        'total_leaked_pairs': (len(leakage['train_val']) + len(leakage['train_test']) + 
                               len(leakage['val_test']) + len(leakage['all_splits'])),
        'train_size': result['train_size'],
        'val_size': result['val_size'],
        'test_size': result['test_size']
    })

leakage_df = pd.DataFrame(leakage_summary)
leakage_df.to_csv('Leakage_Analysis/leakage_summary.csv', index=False)
print("Saved: Leakage_Analysis/leakage_summary.csv")

# Save examples of leaked pairs for first side effect
if leakage['train_test'] or leakage['train_val'] or leakage['all_splits']:
    examples = []
    for norm_pair, info in list(leakage['train_test'].items())[:20]:
        examples.append({
            'drug1': info['drug_names'][0],
            'drug2': info['drug_names'][1],
            'normalized_pair': str(norm_pair),
            'in_train_as': str(info['train']),
            'in_test_as': str(info['test']),
            'leakage_type': 'Train & Test'
        })
    
    for norm_pair, info in list(leakage['train_val'].items())[:20]:
        examples.append({
            'drug1': info['drug_names'][0],
            'drug2': info['drug_names'][1],
            'normalized_pair': str(norm_pair),
            'in_train_as': str(info['train']),
            'in_val_as': str(info['val']),
            'leakage_type': 'Train & Val'
        })
    
    for norm_pair, info in list(leakage['all_splits'].items())[:20]:
        examples.append({
            'drug1': info['drug_names'][0],
            'drug2': info['drug_names'][1],
            'normalized_pair': str(norm_pair),
            'in_train_as': str(info['train']),
            'in_val_as': str(info['val']),
            'in_test_as': str(info['test']),
            'leakage_type': 'All Three Splits'
        })
    
    examples_df = pd.DataFrame(examples)
    examples_df.to_csv('Leakage_Analysis/leaked_pairs_examples.csv', index=False)
    print("Saved: Leakage_Analysis/leaked_pairs_examples.csv")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nAll results saved to 'Leakage_Analysis/' directory:")
print("  - leakage_summary_all_side_effects.png")
print("  - leakage_detailed_side_effect_1.png")
print("  - leakage_summary.csv")
print("  - leaked_pairs_examples.csv")
print("\n" + "=" * 80)
