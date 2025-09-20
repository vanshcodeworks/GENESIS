import os
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import pyranges as pr
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DNA_CNN_MultiClass(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 32, 12), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Conv1d(32, 64, 8), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Flatten(), nn.Linear(640, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.conv_net(x)

def load_model(device):
    with open('models/label_map.json', 'r') as f:
        label_map = json.load(f)  # {"0":"Other",...}
    model = DNA_CNN_MultiClass(num_classes=len(label_map)).to(device)
    state = torch.load('models/genesis_multiclass.pth', map_location=device)
    if isinstance(state, dict) and any(k.startswith('net.') for k in state.keys()):
        state = {k.replace('net.', 'conv_net.'): v for k, v in state.items()}
    model.load_state_dict(state); model.eval()
    inv = {v: int(k) for k, v in label_map.items()}
    return model, label_map, inv

def rebuild_test_split():
    X = np.load('processed_sequences.npy')             # (N, 200, 4)
    X = np.transpose(X, (0, 2, 1))                     # (N, 4, 200)
    gvf_df = pd.read_csv('data/homo_sapiens-chr22.gvf', sep='\t', comment='#',
                         names=['Chromosome','Source','Feature','Start','End','Score','Strand','Frame','Attribute'],
                         dtype={'Chromosome': str})
    gff_df = pd.read_csv('data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3', sep='\t', comment='#',
                         names=['Chromosome','Source','Feature','Start','End','Score','Strand','Frame','Attribute'],
                         dtype={'Chromosome': str})
    reg_features = pr.PyRanges(gff_df); promoters = reg_features[reg_features.Feature == 'promoter']
    variants = pr.PyRanges(gvf_df); results_df = variants.join(promoters).df
    genes = {'CHEK2': {'start': 28686915, 'end': 28741334, 'label': 1},
             'TBX1':  {'start': 19850059, 'end': 19879068, 'label': 2},
             'COMT':  {'start': 19942588, 'end': 19969679, 'label': 3}}
    y = np.zeros(len(results_df), dtype=int)
    for _, info in genes.items():
        y[(results_df['Start'] >= info['start']) & (results_df['End'] <= info['end'])] = info['label']
    if len(y) != len(X): y = y[:len(X)]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_test, y_test

def saliency_for_input(model, device, x4x200, target_idx=None):
    # x4x200: (4,200) float32
    x = torch.tensor(x4x200[None, ...], dtype=torch.float32, device=device, requires_grad=True)
    logits = model(x)
    if target_idx is None:
        target_idx = int(torch.argmax(logits, dim=1).item())
    loss = logits[0, target_idx]
    model.zero_grad()
    loss.backward()
    grad = x.grad.detach().cpu().numpy()[0]  # (4,200)
    # aggregate across channels (max-abs or L2)
    sal = np.max(np.abs(grad), axis=0)       # (200,)
    return sal, target_idx

def plot_saliency(sal, title, out_path):
    plt.figure(figsize=(10, 2))
    plt.imshow(sal[None, :], aspect='auto', cmap='magma')
    plt.yticks([]); plt.xlabel('Position'); plt.title(title)
    os.makedirs(os.path.join('reports','figures'), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    print(f"Saved: {out_path}")

def main(n_per_class=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, label_map, inv = load_model(device)
    X_test, _ = rebuild_test_split()

    # Score test set to select high-confidence samples per class
    with torch.no_grad():
        probs = []
        for i in range(0, len(X_test), 256):
            batch = torch.tensor(X_test[i:i+256], dtype=torch.float32, device=device)
            p = torch.softmax(model(batch), dim=1).cpu().numpy()
            probs.append(p)
        probs = np.vstack(probs)  # (N_test, C)

    classes = [label_map[str(i)] for i in range(probs.shape[1])]
    for ci, cname in enumerate(classes):
        # pick top-N for this class by confidence
        idx_sorted = np.argsort(-probs[:, ci])
        chosen = idx_sorted[:n_per_class]
        sal_stack = []
        for idx in chosen:
            sal, _ = saliency_for_input(model, device, X_test[idx], target_idx=ci)
            sal_stack.append(sal)
        if not sal_stack:
            continue
        sal_avg = np.mean(np.stack(sal_stack, axis=0), axis=0)  # (200,)
        out_avg = os.path.join('reports','figures', f'saliency_avg_{cname}.png')
        plot_saliency(sal_avg, f'Saliency (avg) – {cname}', out_avg)

        # also dump first example
        out_one = os.path.join('reports','figures', f'saliency_example_{cname}.png')
        plot_saliency(sal_stack[0], f'Saliency (example) – {cname}', out_one)

if __name__ == '__main__':
    main()
