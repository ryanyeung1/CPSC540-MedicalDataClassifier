import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from pathlib import Path


def plot_confusion_matrix(y_true, y_preds, classes, model_names, dataset):

    # Generate confusion matrix

    # Create subplots for each model
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Performance of models on {dataset}', fontsize=16)

    for i in range(2):
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_preds[i])

        # Create a heatmap
        sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", xticklabels=classes, yticklabels=classes, ax=axes[i])

        axes[i].set_title(f'Confusion Matrix for {model_names[i]}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')

    savefig(f"{dataset}_cm.jpg", fig=fig)
    plt.close(fig)


def calculate_metrics(y_true, y_pred, average='weighted'):

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    
    return accuracy, precision, recall, f1

def visualize_metrics_bar_chart(metrics_values, models, dataset_types):

    # Set color palette based on dataset type
    colors = sns.color_palette("Set2", n_colors=len(dataset_types))
    
    # Create subplots for each metric
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.suptitle('Metrics Comparison for Different Models', fontsize=16)

    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, metric_name in enumerate(metrics_names):
        row, col = divmod(i, 2)
        ax = axes[row, col]

        # Plot bar chart for each model
        for j, model in enumerate(models):
            ax.bar(np.arange(len(dataset_types)) + j * 0.2, metrics_values[:, j, i], width=0.2, label=f'{model}', color=colors[j])

        ax.set_xlabel('Datasets')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(np.arange(len(dataset_types)) + 0.2)
        ax.set_xticklabels(dataset_types)
        ax.legend()

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    savefig(f"eval_metrics.jpg", fig=fig)
    plt.close(fig)


def savefig(fname, fig=None, verbose=True):
    path = Path(".", fname)
    (plt if fig is None else fig).savefig(path, bbox_inches="tight", pad_inches=1)
    if verbose:
        print(f"Figure saved as '{path}'")

