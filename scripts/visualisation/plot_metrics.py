import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

METRICS_PATH = os.path.join('models', 'metrics.json')


def main():
    if not os.path.exists(METRICS_PATH):
        print('metrics.json not found. Run training first.')
        return
    with open(METRICS_PATH, 'r') as f:
        metrics = json.load(f)

    cm = metrics.get('confusion_matrix')
    if cm is None:
        print('No confusion matrix in metrics.json')
        return
    cm = np.array(cm)

    labels = metrics.get('target_names') or ['Other', 'CHEK2', 'TBX1', 'COMT']

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)
    out = os.path.join('reports', 'figures', 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
