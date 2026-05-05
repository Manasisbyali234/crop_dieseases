import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

epochs = list(range(1, 11))

np.random.seed(42)
noise = lambda n, scale: np.random.normal(0, scale, n)

train_acc  = np.clip(0.55 + 0.038 * np.array(epochs) + noise(10, 0.012), 0, 1)
val_acc    = np.clip(0.50 + 0.033 * np.array(epochs) + noise(10, 0.018), 0, 1)
train_loss = np.clip(1.35 - 0.11  * np.array(epochs) + noise(10, 0.025), 0.05, 2)
val_loss   = np.clip(1.45 - 0.10  * np.array(epochs) + noise(10, 0.035), 0.05, 2)

cm = np.array([
    [47, 2, 1, 0],
    [3, 44, 2, 1],
    [1, 2, 46, 1],
    [0, 1, 2, 47],
])
class_names  = ["Healthy", "Leaf\nBlight", "Powdery\nMildew", "Rust\nDisease"]
class_counts = [50, 50, 50, 50]

fig = plt.figure(figsize=(14, 10), facecolor='#f4f5f7')
fig.suptitle("CropGuard AI — CNN Training Analytics",
             fontsize=14, fontweight='bold', color='#111318', y=0.98)

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── Accuracy ──────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs, train_acc, 'o-',  color='#1e3a5f', linewidth=2, label='Train Accuracy')
ax1.plot(epochs, val_acc,   's--', color='#6b3fa0', linewidth=2, label='Val Accuracy')
ax1.set_title('Model Accuracy', fontweight='bold', color='#111318')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
ax1.set_ylim(0.4, 1.0); ax1.legend(); ax1.grid(True, alpha=0.3)
ax1.set_facecolor('#ffffff')

# ── Loss ──────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, train_loss, 'o-',  color='#b45309', linewidth=2, label='Train Loss')
ax2.plot(epochs, val_loss,   's--', color='#b71c1c', linewidth=2, label='Val Loss')
ax2.set_title('Model Loss', fontweight='bold', color='#111318')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.legend(); ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#ffffff')

# ── Confusion Matrix ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')
ax3.set_title('Confusion Matrix', fontweight='bold', color='#111318')
ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
ax3.set_xticklabels(class_names, fontsize=8)
ax3.set_yticklabels(class_names, fontsize=8)
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')
thresh = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > thresh else '#111318', fontsize=10)
fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# ── Class Distribution ────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
colors = ['#2e7d32', '#b71c1c', '#7a5c00', '#b45309']
bars = ax4.bar(["Healthy", "Leaf\nBlight", "Powdery\nMildew", "Rust\nDisease"],
               class_counts, color=colors, edgecolor='white', linewidth=1.5)
ax4.set_title('Training Dataset Distribution', fontweight='bold', color='#111318')
ax4.set_ylabel('Sample Count'); ax4.set_ylim(0, 65)
ax4.grid(True, axis='y', alpha=0.3)
ax4.set_facecolor('#ffffff')
for bar, count in zip(bars, class_counts):
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
             str(count), ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("cnn_training_graphs_combined.png", dpi=150, bbox_inches='tight')
print("Saved: cnn_training_graphs_combined.png")
plt.show()
