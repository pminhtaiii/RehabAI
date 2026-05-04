import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plots directory exists
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set global styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12})

def generate_mock_scatter_data(exercises, augmentations, n_samples=100):
    """Generates mock evaluation data for ground truth vs predictions."""
    data = []
    for ex in exercises:
        for aug in augmentations:
            # Ground truth scores between 0 and 100
            y_true = np.random.uniform(0, 100, n_samples)
            # Predictions add some noise. Augmented models are simulated to be more accurate.
            noise_scale = 8 if aug == "Augmented" else 15
            y_pred = y_true + np.random.normal(0, noise_scale, n_samples)
            # Clip between 0 and 100
            y_pred = np.clip(y_pred, 0, 100)
            
            df = pd.DataFrame({
                'Exercise': ex,
                'Augmentation': aug,
                'Ground_Truth': y_true,
                'Predicted': y_pred
            })
            data.append(df)
    return pd.concat(data, ignore_index=True)

def plot_scatter_predictions(df):
    """
    Creates a 2x5 grid of scatter plots for Original vs Augmented models
    across exercises Es1 to Es5.
    """
    exercises = ["Es1", "Es2", "Es3", "Es4", "Es5"]
    augmentations = ["Original", "Augmented"]
    
    fig, axes = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey=True)
    
    for i, aug in enumerate(augmentations):
        for j, ex in enumerate(exercises):
            ax = axes[i, j]
            subset = df[(df['Exercise'] == ex) & (df['Augmentation'] == aug)]
            
            # Use different colors for Original vs Augmented for visual distinction
            color = '#1f77b4' if aug == "Original" else '#2ca02c'
            
            sns.scatterplot(
                data=subset, x='Ground_Truth', y='Predicted', 
                ax=ax, alpha=0.6, color=color, edgecolor='w', s=50
            )
            
            # Perfect prediction line (y = x)
            ax.plot([0, 100], [0, 100], 'r--', lw=2, alpha=0.8)
            
            if i == 0:
                ax.set_title(ex, fontsize=16, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{aug}\nPredicted Score', fontsize=14)
            else:
                ax.set_ylabel('')
            if i == 1:
                ax.set_xlabel('Ground Truth Score', fontsize=14)
            else:
                ax.set_xlabel('')
                
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            
    plt.suptitle('Predicted vs Ground-Truth Clinical Scores', fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'scatter_predictions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {os.path.join(PLOTS_DIR, 'scatter_predictions.png')}")

def plot_loss_curves():
    """Generates loss curves for Training vs Validation mock data."""
    epochs = np.arange(1, 51)
    # Mock loss data
    train_loss = np.exp(-0.1 * epochs) + np.random.normal(0, 0.02, 50)
    val_loss = np.exp(-0.08 * epochs) + np.random.normal(0, 0.05, 50) + 0.1
    
    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_loss, label='Train Loss', lw=2)
    plt.plot(epochs, val_loss, label='Validation Loss', lw=2)
    plt.title('Training and Validation Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (MSE)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'loss_curves.png'), dpi=300)
    plt.close()
    print(f"Saved {os.path.join(PLOTS_DIR, 'loss_curves.png')}")

def plot_correlation_bars():
    """Generates a bar chart showing feature vs target correlation."""
    features = ['Shoulder\nAngle', 'Elbow\nAngle', 'Hip\nAngle', 'Knee\nAngle', 'Ankle\nAngle']
    correlations = [0.85, 0.78, 0.65, 0.55, 0.45]
    
    plt.figure(figsize=(9, 6))
    sns.barplot(x=features, y=correlations, hue=features, palette='viridis', legend=False)
    plt.title('Feature Correlation with Clinical Score', fontsize=16)
    plt.ylabel('Pearson Correlation Coefficient', fontsize=14)
    plt.ylim(0, 1)
    plt.xticks(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'correlation_bars.png'), dpi=300)
    plt.close()
    print(f"Saved {os.path.join(PLOTS_DIR, 'correlation_bars.png')}")

def main():
    print(f"Creating plots in '{PLOTS_DIR}' directory...")
    
    # 1. Scatter Plots (Predicted vs Ground Truth for Es1-Es5, with and without augmentation)
    exercises = ["Es1", "Es2", "Es3", "Es4", "Es5"]
    augmentations = ["Original", "Augmented"]
    
    # In a real scenario, you would read an actual evaluations CSV file:
    # df = pd.read_csv('evaluation_results.csv')
    print("Generating mock scatter data...")
    df_scatter = generate_mock_scatter_data(exercises, augmentations)
    plot_scatter_predictions(df_scatter)
    
    # 2. Loss Curves
    print("Generating loss curves...")
    plot_loss_curves()
    
    # 3. Correlation bar charts
    print("Generating correlation bar chart...")
    plot_correlation_bars()
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
