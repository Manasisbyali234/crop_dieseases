"""
Generate CNN Training Graphs and save them as image files
Run this script to create graph images in the project folder
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Simulated training data
np.random.seed(42)
epochs = list(range(1, 11))

noise = lambda n, scale: np.random.normal(0, scale, n)

train_acc = np.clip(0.55 + 0.038 * np.array(epochs) + noise(10, 0.012), 0, 1)
val_acc   = np.clip(0.50 + 0.033 * np.array(epochs) + noise(10, 0.018), 0, 1)
train_loss = np.clip(1.35 - 0.11 * np.array(epochs) + noise(10, 0.025), 0.05, 2)
val_loss   = np.clip(1.45 - 0.10 * np.array(epochs) + noise(10, 0.035), 0.05, 2)

# Confusion matrix
cm = np.array([
    [47, 2, 1, 0],
    [3, 44, 2, 1],
    [1, 2, 46, 1],
    [0, 1, 2, 47],
])
class_names = ["Healthy", "Leaf\nBlight", "Powdery\nMildew", "Rust\nDisease"]
class_counts = [50, 50, 50, 50]

# ═══════════════════════════════════════════════════════════════════════════
# COMBINED GRAPH (All 4 graphs in one image)
# ═══════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(14, 10), facecolor='#f4f5f7')
fig.suptitle("CropGuard AI — CNN Training Analytics",
             fontsize=16, fontweight='bold', color='#111318', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Accuracy ──
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_acc, 'o-', color='#1e3a5f', linewidth=2.5, markersize=6, label='Train Accuracy')
ax1.plot(epochs, val_acc,   's--', color='#6b3fa0', linewidth=2.5, markersize=6, label='Val Accuracy')
ax1.set_title('Model Accuracy', fontweight='bold', fontsize=12, color='#111318', pad=10)
ax1.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=10, fontweight='bold')
ax1.set_ylim(0.4, 1.0)
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_facecolor('#ffffff')

# ── Loss ──
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_loss, 'o-', color='#b45309', linewidth=2.5, markersize=6, label='Train Loss')
ax2.plot(epochs, val_loss,   's--', color='#b71c1c', linewidth=2.5, markersize=6, label='Val Loss')
ax2.set_title('Model Loss', fontweight='bold', fontsize=12, color='#111318', pad=10)
ax2.set_xlabel('Epoch', fontsize=10, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=10, fontweight='bold')
ax2.legend(fontsize=9, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_facecolor('#ffffff')

# ── Confusion Matrix ──
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title('Confusion Matrix', fontweight='bold', fontsize=12, color='#111318', pad=10)
ax3.set_xticks(range(4))
ax3.set_yticks(range(4))
ax3.set_xticklabels(class_names, fontsize=9)
ax3.set_yticklabels(class_names, fontsize=9)
ax3.set_xlabel('Predicted', fontsize=10, fontweight='bold')
ax3.set_ylabel('Actual', fontsize=10, fontweight='bold')
thresh = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else '#111318', 
                 fontsize=11, fontweight='bold')
fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# ── Class Distribution ──
ax4 = fig.add_subplot(gs[1, 1])
colors = ['#2e7d32', '#b71c1c', '#7a5c00', '#b45309']
bars = ax4.bar(["Healthy", "Leaf\nBlight", "Powdery\nMildew", "Rust\nDisease"],
               class_counts, color=colors, edgecolor='white', linewidth=2)
ax4.set_title('Training Dataset Distribution', fontweight='bold', fontsize=12, color='#111318', pad=10)
ax4.set_ylabel('Sample Count', fontsize=10, fontweight='bold')
ax4.set_ylim(0, 65)
ax4.grid(True, axis='y', alpha=0.3, linestyle='--')
ax4.set_facecolor('#ffffff')
for bar, count in zip(bars, class_counts):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('cnn_training_graphs_combined.png', dpi=300, bbox_inches='tight', facecolor='#f4f5f7')
print("[OK] Saved: cnn_training_graphs_combined.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# INDIVIDUAL GRAPHS
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Accuracy Graph ──
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f4f5f7')
ax.plot(epochs, train_acc, 'o-', color='#1e3a5f', linewidth=3, markersize=8, label='Train Accuracy')
ax.plot(epochs, val_acc,   's--', color='#6b3fa0', linewidth=3, markersize=8, label='Validation Accuracy')
ax.set_title('CNN Model Accuracy Over Epochs', fontweight='bold', fontsize=14, color='#111318', pad=15)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#ffffff')
plt.tight_layout()
plt.savefig('cnn_accuracy_graph.png', dpi=300, bbox_inches='tight', facecolor='#f4f5f7')
print("[OK] Saved: cnn_accuracy_graph.png")
plt.close()

# ── 2. Loss Graph ──
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f4f5f7')
ax.plot(epochs, train_loss, 'o-', color='#b45309', linewidth=3, markersize=8, label='Train Loss')
ax.plot(epochs, val_loss,   's--', color='#b71c1c', linewidth=3, markersize=8, label='Validation Loss')
ax.set_title('CNN Model Loss Over Epochs', fontweight='bold', fontsize=14, color='#111318', pad=15)
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#ffffff')
plt.tight_layout()
plt.savefig('cnn_loss_graph.png', dpi=300, bbox_inches='tight', facecolor='#f4f5f7')
print("[OK] Saved: cnn_loss_graph.png")
plt.close()

# ── 3. Confusion Matrix ──
fig, ax = plt.subplots(figsize=(8, 7), facecolor='#f4f5f7')
im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
ax.set_title('Confusion Matrix - Disease Classification', fontweight='bold', fontsize=14, color='#111318', pad=15)
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(class_names, fontsize=11)
ax.set_yticklabels(class_names, fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
thresh = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                color='white' if cm[i, j] > thresh else '#111318', 
                fontsize=13, fontweight='bold')
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Count', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('cnn_confusion_matrix.png', dpi=300, bbox_inches='tight', facecolor='#f4f5f7')
print("[OK] Saved: cnn_confusion_matrix.png")
plt.close()

# ── 4. Class Distribution ──
fig, ax = plt.subplots(figsize=(10, 6), facecolor='#f4f5f7')
colors = ['#2e7d32', '#b71c1c', '#7a5c00', '#b45309']
bars = ax.bar(["Healthy", "Leaf Blight", "Powdery Mildew", "Rust Disease"],
              class_counts, color=colors, edgecolor='white', linewidth=2.5, width=0.6)
ax.set_title('Training Dataset Class Distribution', fontweight='bold', fontsize=14, color='#111318', pad=15)
ax.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax.set_ylim(0, 65)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_facecolor('#ffffff')
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig('cnn_class_distribution.png', dpi=300, bbox_inches='tight', facecolor='#f4f5f7')
print("[OK] Saved: cnn_class_distribution.png")
plt.close()

print("\n" + "="*60)
print("All CNN training graphs generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. cnn_training_graphs_combined.png  (All 4 graphs)")
print("  2. cnn_accuracy_graph.png            (Accuracy only)")
print("  3. cnn_loss_graph.png                (Loss only)")
print("  4. cnn_confusion_matrix.png          (Confusion matrix)")
print("  5. cnn_class_distribution.png        (Dataset distribution)")
print("\nThese images are ready to include in your documentation.")
