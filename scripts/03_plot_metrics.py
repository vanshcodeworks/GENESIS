import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score  # NEW

METRICS_PATH = os.path.join('models', 'metrics.json')


def _ensure_outdir():
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)


def _plot_roc_ovr(y_true, y_score, class_names):
    # y_true: (N,), y_score: (N, C)
    n_classes = y_score.shape[1]
    plt.figure(figsize=(7, 6))
    # micro-average
    fpr_micro, tpr_micro, _ = roc_curve(np.eye(n_classes)[y_true].ravel(), y_score.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='black', lw=2, label=f'micro-average (AUC={auc_micro:.3f})')
    # per-class
    for i in range(n_classes):
        y_true_i = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_i, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1.5, label=f'{class_names[i]} (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves (One-vs-Rest)')
    plt.legend(fontsize=8, loc='lower right')
    _ensure_outdir()
    out = os.path.join('reports', 'figures', 'roc_ovr.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def _plot_pr_ovr(y_true, y_score, class_names):
    n_classes = y_score.shape[1]
    plt.figure(figsize=(7, 6))
    # micro-average
    precision_micro, recall_micro, _ = precision_recall_curve(np.eye(n_classes)[y_true].ravel(), y_score.ravel())
    ap_micro = average_precision_score(np.eye(n_classes)[y_true].ravel(), y_score.ravel())
    plt.plot(recall_micro, precision_micro, color='black', lw=2, label=f'micro-average (AP={ap_micro:.3f})')
    # per-class
    for i in range(n_classes):
        y_true_i = (y_true == i).astype(int)
        precision, recall, _ = precision_recall_curve(y_true_i, y_score[:, i])
        ap = average_precision_score(y_true_i, y_score[:, i])
        plt.plot(recall, precision, lw=1.5, label=f'{class_names[i]} (AP={ap:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curves (One-vs-Rest)')
    plt.legend(fontsize=8, loc='lower left')
    _ensure_outdir()
    out = os.path.join('reports', 'figures', 'pr_ovr.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f'Saved: {out}')


def main():
    if not os.path.exists(METRICS_PATH):
        print('metrics.json not found. Run training first.')
        return
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    # Confusion matrix
    cm = metrics.get('confusion_matrix')
    if cm is not None:
        cm = np.array(cm)
        labels = metrics.get('target_names') or ['Other', 'CHEK2', 'TBX1', 'COMT']
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        _ensure_outdir()
        out = os.path.join('reports', 'figures', 'confusion_matrix.png')
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close()
        print(f'Saved: {out}')
    else:
        print('No confusion matrix in metrics.json')

    # ROC and PR (require saved scores)
    probs_path = os.path.join('models', 'test_probs.npy')
    y_path = os.path.join('models', 'test_labels.npy')
    if os.path.exists(probs_path) and os.path.exists(y_path):
        y_score = np.load(probs_path)
        y_true = np.load(y_path)
        class_names = metrics.get('target_names') or ['Other', 'CHEK2', 'TBX1', 'COMT'][:y_score.shape[1]]
        _plot_roc_ovr(y_true, y_score, class_names)
        _plot_pr_ovr(y_true, y_score, class_names)
    else:
        print('test_probs.npy and/or test_labels.npy not found. Re-run training to generate them.')


if __name__ == '__main__':
    main()
