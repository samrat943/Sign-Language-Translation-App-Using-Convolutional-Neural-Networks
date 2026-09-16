import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Generate a clean ASL Alphabet Reference Chart Image for the UI
chart_path = os.path.join("app", "asl_chart.png")
processed_dir = os.path.join("data", "processed")

label_map_path = os.path.join(processed_dir, "label_map.json")
if os.path.exists(label_map_path):
    with open(label_map_path, "r") as f:
        label_map = json.load(f)
else:
    label_map = {str(i): chr(65 + (i if i < 9 else i + 1)) for i in range(24)}

fig, axes = plt.subplots(4, 6, figsize=(12, 8))
fig.suptitle("ASL Sign Language Model Supported Gestures (A–I, K–Y)", fontsize=16, fontweight="bold", y=0.98)

# Load sample image data if available
x_test_path = os.path.join(processed_dir, "test_x.npy")
y_test_path = os.path.join(processed_dir, "test_y.npy")

if os.path.exists(x_test_path) and os.path.exists(y_test_path):
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)
    
    for idx in range(24):
        row, col = idx // 6, idx % 6
        ax = axes[row, col]
        
        # Find first sample matching this class index
        sample_indices = np.where(y_test == idx)[0]
        if len(sample_indices) > 0:
            sample_img = x_test[sample_indices[0]].squeeze()
            ax.imshow(sample_img, cmap='gray')
        else:
            ax.text(0.5, 0.5, label_map.get(str(idx), str(idx)), fontsize=20, ha='center')
            
        ax.set_title(f"Class {idx}: '{label_map.get(str(idx), str(idx))}'", fontsize=11, fontweight="bold")
        ax.axis('off')
else:
    for idx in range(24):
        row, col = idx // 6, idx % 6
        ax = axes[row, col]
        letter = label_map.get(str(idx), str(idx))
        ax.text(0.5, 0.5, letter, fontsize=32, ha='center', va='center', fontweight='bold', color='#00ff88')
        ax.set_title(f"Letter '{letter}'", fontsize=11, fontweight="bold")
        ax.set_facecolor('#161b22')
        ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Generated ASL reference chart image at '{chart_path}'")
