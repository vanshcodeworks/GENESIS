# debug_prediction.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyranges as pr
import json
from sklearn.model_selection import train_test_split
import argparse
from pathlib import Path

# --- Re-define the Model Architecture (Must match the trained model) ---
class DNA_CNN_MultiClass(nn.Module):
    def __init__(self, num_classes=4):
        super(DNA_CNN_MultiClass, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=12),
            nn.ReLU(), nn.MaxPool1d(kernel_size=4), nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8),
            nn.ReLU(), nn.MaxPool1d(kernel_size=4), nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_features=640, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )
    def forward(self, x):
        return self.conv_net(x)

def sanitize(seq: str, length: int = 200) -> str:
    s = (seq or '').upper().replace('\n','').replace('\r','')
    s = ''.join(ch if ch in 'ACGT' else 'A' for ch in s)
    if len(s) > length: s = s[:length]
    if len(s) < length: s = s + 'A' * (length - len(s))
    return s

def topk(probs: np.ndarray, label_map: dict, k: int = 3):
    idx = np.argsort(-probs)[:k]
    return [(label_map[str(i)], float(probs[i])) for i in idx]

def decode_one_hot_to_acgt(arr):
    """
    arr: (4, 200) or (200, 4) one-hot-like array -> 200bp string
    """
    a = arr
    if a.shape[0] == 200 and a.shape[1] == 4:
        a = a.T
    idx = np.argmax(a, axis=0)  # (200,)
    alphabet = np.array(list('ACGT'))
    return ''.join(alphabet[idx])

def load_model_and_labels(device):
    with open('models/label_map.json', 'r') as f:
        label_map = json.load(f)
    model = DNA_CNN_MultiClass(num_classes=len(label_map))
    model.load_state_dict(torch.load('models/genesis_multiclass.pth', map_location=device))
    model.to(device).eval()
    return model, label_map

def run_known_sample_check(device):
    print("--- Starting Model Integrity Check ---")
    print(f"Using device: {device}")
    print("-" * 50)
    X_np = np.load('processed_sequences.npy')
    X_np = np.transpose(X_np, (0, 2, 1))
    gvf_df = pd.read_csv('data/homo_sapiens-chr22.gvf', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})
    gff_df = pd.read_csv('data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})
    reg_features = pr.PyRanges(gff_df); promoters = reg_features[reg_features.Feature == 'promoter']
    variants = pr.PyRanges(gvf_df); results_df = variants.join(promoters).df
    genes = {'CHEK2': {'start': 28686915, 'end': 28741334, 'label': 1},
             'TBX1': {'start': 19850059, 'end': 19879068, 'label': 2},
             'COMT': {'start': 19942588, 'end': 19969679, 'label': 3}}
    y_np = np.zeros(len(results_df), dtype=int)
    for gene, info in genes.items():
        in_gene = (results_df['Start'] >= info['start']) & (results_df['End'] <= info['end'])
        y_np[in_gene] = info['label']
    if len(y_np) != len(X_np): y_np = y_np[:len(X_np)]
    _, X_test, _, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)
    print("✓ Recreated the original test set.")

    model, label_map = load_model_and_labels(device)
    idxs = np.where(y_test == 1)[0]
    if len(idxs) == 0:
        print("✗ Error: No CHEK2 samples found in the recreated test set.")
        return
    x = torch.tensor(X_test[idxs[0]][None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    print("Known-sample check (true CHEK2):", topk(p, label_map, k=4))

def run_free_text(device, seq: str):
    model, label_map = load_model_and_labels(device)
    seq = sanitize(seq, 200)
    oh = np.zeros((200, 4), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in 'ACGT':
            oh[i, 'ACGT'.index(b)] = 1.0
    x = torch.tensor(oh.T[None, ...], dtype=torch.float32).to(device)
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    print("Free-text input top-k:", topk(p, label_map, k=4))

def export_examples(device, out_path: str, per_class: int = 10, min_conf: float = 0.90):
    """
    Export curated 200bp sequences per class from the test split into a JSON file
    that the demo can consume. Sequences are chosen where model predicts that class
    with confidence >= min_conf.
    """
    print(f"Exporting curated examples to {out_path} (per_class={per_class}, min_conf={min_conf})")
    X_np = np.load('processed_sequences.npy')  # (N, 200, 4)
    X_np = np.transpose(X_np, (0, 2, 1))       # (N, 4, 200)

    gvf_df = pd.read_csv('data/homo_sapiens-chr22.gvf', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})
    gff_df = pd.read_csv('data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})
    reg_features = pr.PyRanges(gff_df); promoters = reg_features[reg_features.Feature == 'promoter']
    variants = pr.PyRanges(gvf_df); results_df = variants.join(promoters).df
    genes = {'CHEK2': {'start': 28686915, 'end': 28741334, 'label': 1},
             'TBX1': {'start': 19850059, 'end': 19879068, 'label': 2},
             'COMT': {'start': 19942588, 'end': 19969679, 'label': 3}}
    y_np = np.zeros(len(results_df), dtype=int)
    for gene, info in genes.items():
        in_gene = (results_df['Start'] >= info['start']) & (results_df['End'] <= info['end'])
        y_np[in_gene] = info['label']
    if len(y_np) != len(X_np): y_np = y_np[:len(X_np)]

    _, X_test, _, y_test = train_test_split(X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

    model, label_map = load_model_and_labels(device)
    inv_label = {v: int(k) for k, v in label_map.items()}

    # Collect examples
    buckets = {name: [] for name in label_map.values()}
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            x = torch.tensor(X_test[i][None, ...], dtype=torch.float32).to(device)
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_name = label_map[str(pred_idx)]
            conf = float(probs[pred_idx])
            if conf >= min_conf and len(buckets[pred_name]) < per_class:
                seq = decode_one_hot_to_acgt(X_test[i])
                buckets[pred_name].append(seq)
            # stop early if all filled
            if all(len(v) >= per_class for v in buckets.values()):
                break

    # Ensure keys exist
    for k in list(buckets.keys()):
        if k not in ['Other', 'CHEK2', 'TBX1', 'COMT']:
            buckets.pop(k, None)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(buckets, f, indent=2)
    print(f"✓ Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='Run known-sample integrity check')
    ap.add_argument('--seq', help='Free-text 200bp sequence to test')
    ap.add_argument('--export-examples', metavar='PATH', help='Write curated demo examples JSON to PATH (e.g., demo/examples.json)')
    ap.add_argument('--per-class', type=int, default=10, help='Examples per class to export')
    ap.add_argument('--min-conf', type=float, default=0.90, help='Minimum confidence for selecting examples')
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.check:
        run_known_sample_check(device)
    if args.seq:
        run_free_text(device, args.seq)
    if args.export_examples:
        export_examples(device, args.export_examples, per_class=args.per_class, min_conf=args.min_conf)
    if not args.check and not args.seq and not args.export_examples:
        print("Usage:")
        print("  python debug_prediction.py --check")
        print("  python debug_prediction.py --seq ACGT...")
        print("  python debug_prediction.py --export-examples demo/examples.json --per-class 6 --min-conf 0.95")

if __name__ == "__main__":
    main()