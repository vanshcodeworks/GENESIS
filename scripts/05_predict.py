import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    import pyfaidx  
except Exception:
    pyfaidx = None

class DNA_CNN_MultiClass(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, 32, 12), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Conv1d(32, 64, 8), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(0.2),
            nn.Flatten(), nn.Linear(640, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.conv_net(x)

NUC = {'A':0,'C':1,'G':2,'T':3}

def one_hot(seq):
    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for i, b in enumerate(seq.upper()):
        if b in NUC:
            arr[i, NUC[b]] = 1.0
    return arr

def sanitize(seq: str, length: int = 200) -> str:
    s = (seq or '').upper().replace('\n','').replace('\r','')
    s = ''.join(ch if ch in 'ACGT' else 'A' for ch in s)
    if len(s) > length: s = s[:length]
    if len(s) < length: s = s + 'A' * (length - len(s))
    return s


def fetch_region_sequence(chrom: str, pos: int, fasta_path: str, flank: int = 100) -> str:
    if pyfaidx is None:
        raise RuntimeError('pyfaidx not installed. Install it or pass --seq.')
    genome = pyfaidx.Fasta(fasta_path)
    start = max(0, pos - flank)
    end = pos + flank
    return genome[str(chrom)][start:end].seq


def predict_sequence(seq, model_path='models/genesis_multiclass.pth', label_map_path='models/label_map.json', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with open(label_map_path, 'r') as f:
        label_map = {int(k): v for k, v in json.load(f).items()}
    num_classes = max(label_map.keys()) + 1
    model = DNA_CNN_MultiClass(num_classes=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith('net.') for k in state.keys()):
        state = {k.replace('net.', 'conv_net.'): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    X = one_hot(seq)
    if X.shape[0] != 200:
        raise ValueError('Sequence length must be 200 bp')
    X = np.transpose(X, (1, 0))  # (4, L)
    X = torch.tensor(X[None, ...], dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return {
            'prediction_index': pred_idx,
            'prediction_label': label_map.get(pred_idx, str(pred_idx)),
            'probabilities': {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)}
        }


def main():
    ap = argparse.ArgumentParser(description='Predict class for a 200bp DNA sequence or genomic region')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--seq', help='Input DNA sequence (200bp)')
    g.add_argument('--chrom', help='Chromosome for region-based prediction, e.g., 22')
    ap.add_argument('--pos', type=int, help='1-based position for region-based prediction (center)')
    ap.add_argument('--fasta', default=os.path.join('data', 'Homo_sapiens.GRCh38.dna.chromosome.22.fa'), help='Path to chr22 FASTA')
    ap.add_argument('--model', default='models/genesis_multiclass.pth', help='Path to .pth model')
    ap.add_argument('--labels', default='models/label_map.json', help='Path to label_map.json')
    args = ap.parse_args()

    if args.chrom:
        if args.pos is None:
            raise SystemExit('--pos is required with --chrom')
        seq = fetch_region_sequence(args.chrom, args.pos, args.fasta)
    else:
        seq = args.seq

    seq = sanitize(seq, 200)
    out = predict_sequence(seq, args.model, args.labels)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()