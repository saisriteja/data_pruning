import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image

# Load data from CSV file
file_path = 'detection_results.csv'
df = pd.read_csv(file_path)

# Load the image (assuming all rows have the same image path)
image_path = df['image_path'][0]
image = Image.open(image_path)

# Sort data by confidence and group into 5 confidence levels
df_sorted = df.sort_values(by='confidence', ascending=False)
conf_levels = [0.9, 0.8, 0.7, 0.6, 0.5]

# Create a 5-column grid
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
fig.suptitle("Bounding Boxes at Different Confidence Levels")

# Function to plot bounding boxes
def plot_boxes(ax, image, boxes_df, conf_threshold, title):
    ax.imshow(image)
    for _, row in boxes_df.iterrows():
        if row['confidence'] >= conf_threshold:
            # Draw bounding box
            rect = patches.Rectangle(
                (row['x1'], row['y1']),
                row['x2'] - row['x1'],
                row['y2'] - row['y1'],
                linewidth=2,
                edgecolor='red' if row['confidence'] > 0.7 else 'yellow',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(row['x1'], row['y1'] - 5, f"{row['confidence']:.2f}", 
                    color='white', fontsize=8, backgroundcolor='black')
    ax.set_title(title)
    ax.axis('off')

# Plot bounding boxes for 5 confidence thresholds
for i, conf in enumerate(conf_levels):
    subset = df_sorted[df_sorted['confidence'] >= conf]
    title = f"Conf ≥ {conf}"
    plot_boxes(axes[i], image, subset, conf, title)

# Adjust layout and save plot
plt.tight_layout()
output_file = "confidence_grid_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved as {output_file}")
