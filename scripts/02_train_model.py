import numpy as np
import pandas as pd
import pyranges as pr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import os
import json
from sklearn.utils.class_weight import compute_class_weight

# --- Binary CNN for fallback (outputs raw logit) ---
class DNA_CNN(nn.Module):
    def __init__(self):
        super(DNA_CNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_features=640, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1)
        )

    def forward(self, x):
        return self.conv_net(x)

# --- 1. Define the CNN Model (No Sigmoid at the end) ---
class DNA_CNN_MultiClass(nn.Module):
    def __init__(self, num_classes=4):
        super(DNA_CNN_MultiClass, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=32, kernel_size=12),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_features=640, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        return self.conv_net(x)

def main():
    print("--- Starting CHEK2 Variant Classification Pipeline (Multi-class ready) ---")

    # --- 1. Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU found! Using device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU not found. Using CPU.")
    print("-" * 50)

    # --- 2. Load Sequence Data ---
    print("Loading processed sequence data (X)...")
    try:
        X_np = np.load('processed_sequences.npy')
        print(f"✓ Loaded sequences shape: {X_np.shape}")
    except FileNotFoundError:
        print("✗ Error: 'processed_sequences.npy' not found. Please run scripts/01_prepare_model.py first.")
        return

    # Permute dimensions for PyTorch Conv1d
    print("Permuting dimensions for PyTorch Conv1d...")
    X_np = np.transpose(X_np, (0, 2, 1))
    print(f"✓ Sequences shape after permutation: {X_np.shape}")
    print("-" * 50)

    # --- 3. Generate Biological Labels (aligned to X order if variant_positions.npy exists) ---
    print("Generating multi-class labels for CHEK2, TBX1, COMT (others -> class 0)...")
    gvf_df = pd.read_csv('data/homo_sapiens-chr22.gvf', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})
    gff_df = pd.read_csv('data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3', sep='\t', comment='#',
                         names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                         dtype={'Chromosome': str})

    reg_features = pr.PyRanges(gff_df)
    promoters = reg_features[reg_features.Feature == 'promoter']
    variants = pr.PyRanges(gvf_df)
    variants_in_promoters = variants.join(promoters)
    results_df = variants_in_promoters.df.astype({'Start': int, 'End': int})
    results_df['Chromosome'] = results_df['Chromosome'].astype(str)

    genes = {
        'CHEK2': {'start': 28686915, 'end': 28741334, 'label': 1},
        'TBX1':  {'start': 19850059, 'end': 19879068, 'label': 2},
        'COMT':  {'start': 19942588, 'end': 19969679, 'label': 3}
    }

    # Build mapping from (Chromosome, Start, End) to class label
    print("Building label map by locus...")
    results_df = results_df.copy()
    results_df['label'] = 0
    for _, info in genes.items():
        in_gene = (results_df['Start'] >= info['start']) & (results_df['End'] <= info['end'])
        results_df.loc[in_gene, 'label'] = info['label']
    key_series = list(zip(results_df['Chromosome'].astype(str), results_df['Start'].astype(int), results_df['End'].astype(int)))
    key_to_label = dict(zip(key_series, results_df['label'].astype(int)))

    # Align y to X order if we have persisted variant positions
    if os.path.exists('variant_positions.npy'):
        print("✓ Aligning labels to X order using variant_positions.npy")
        var_pos = np.load('variant_positions.npy', allow_pickle=True)
        y_np = np.zeros(len(var_pos), dtype=int)
        miss = 0
        for i, tup in enumerate(var_pos):
            key = (str(tup[0]), int(tup[1]), int(tup[2]))
            y_np[i] = key_to_label.get(key, 0)
            if key not in key_to_label:
                miss += 1
        if miss:
            print(f"Note: {miss} of {len(var_pos)} keys not found in join; defaulted to Other.")
    else:
        print("variant_positions.npy not found; falling back to unordered label generation (may misalign).")
        y_np = results_df['label'].to_numpy()
        if len(y_np) != len(X_np):
            y_np = y_np[:len(X_np)]

    # Print class counts
    class_counts = np.bincount(y_np, minlength=4)
    class_names = ['Other', 'CHEK2', 'TBX1', 'COMT']
    for idx, name in enumerate(class_names):
        print(f"  Class {idx} ({name}): {class_counts[idx]:,}")
    print("-" * 50)

    # --- 4. Data Splitting and Preparation ---
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=0.2, random_state=42, stratify=y_np)

    print(f"✓ Train set: {X_train.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")
    print("-" * 50)

    # Determine number of classes present (ensure at least 2)
    num_classes = int(np.max(y_train)) + 1
    num_classes = max(num_classes, 2)

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Choose label dtype and criterion by task type
    is_multiclass = num_classes > 2
    if is_multiclass:
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        # Class weights (balanced) across all classes 0..num_classes-1
        counts = np.bincount(y_train, minlength=num_classes).astype(float)
        # Avoid division by zero; zero count -> weight 0
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = (len(y_train) / (num_classes * counts))
            weights[~np.isfinite(weights)] = 0.0
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    else:
        # Binary fallback (CHEK2 vs Other)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        num_positives = np.sum(y_train == 1)
        num_negatives = np.sum(y_train == 0)
        pos_weight = torch.tensor(num_negatives / max(num_positives, 1), dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(f"✓ Created DataLoader with {len(train_loader):,} batches")
    print("-" * 50)

    # --- 5. Initialize Model and Training Components ---
    if is_multiclass:
        model = DNA_CNN_MultiClass(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"✓ Using CrossEntropyLoss with class weights: {weights}")
    else:
        model = DNA_CNN().to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"✓ Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.2f}")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {total_params:,} total ({trainable_params:,} trainable)")
    print(f"✓ Optimizer: Adam with learning rate 0.001")
    print("-" * 50)

    # --- 6. Training Loop with Progress Tracking ---
    print("Training the model...")
    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels if is_multiclass else labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 10 == 0:
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}'})
        epoch_loss = running_loss / len(train_loader)
        print(f"✓ Epoch {epoch+1}/{epochs} completed. Average Loss: {epoch_loss:.4f}")

    print("Model training complete.")
    print("-" * 50)

    # --- 7. Evaluation ---
    print("Evaluating the model...")
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        test_data_loader = DataLoader(TensorDataset(X_test_tensor), batch_size=128)
        for inputs in tqdm(test_data_loader, desc="Evaluating"):
            logits = model(inputs[0].to(device))
            if is_multiclass:
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
            else:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy().ravel())

    print("\nClassification Report:")
    from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score
    metrics_out = {}
    if is_multiclass:
        target_names = ["Other", "CHEK2", "TBX1", "COMT"][0:num_classes]
        report_text = classification_report(y_test, all_preds, target_names=target_names, zero_division=0)
        print(report_text)
        acc = accuracy_score(y_test, all_preds)
        macro_f1 = f1_score(y_test, all_preds, average='macro')
        metrics_out.update({"accuracy": acc, "macro_f1": macro_f1})
        try:
            macro_auc = roc_auc_score(y_test, np.array(all_probs), multi_class='ovr')
            metrics_out["macro_auc_ovr"] = macro_auc
            print(f"✓ Macro AUC (OVR): {macro_auc:.4f}")
        except Exception as e:
            metrics_out["macro_auc_ovr"] = None
            print(f"AUC not available: {e}")
        cm = confusion_matrix(y_test, all_preds)
        print("Confusion Matrix:")
        print(cm)
        metrics_out["confusion_matrix"] = cm.tolist()
        metrics_out["target_names"] = target_names
    else:
        report_text = classification_report(y_test, all_preds, target_names=['Outside CHEK2', 'Inside CHEK2'], zero_division=0)
        print(report_text)
        try:
            auc_val = roc_auc_score(y_test, np.array(all_probs))
            metrics_out["auc_roc"] = auc_val
            print(f"✓ AUC-ROC: {auc_val:.4f}")
        except Exception as e:
            metrics_out["auc_roc"] = None
            print(f"AUC not available: {e}")
        tn, fp, fn, tp = confusion_matrix(y_test, all_preds).ravel()
        print(f"\nConfusion Matrix Breakdown:")
        print(f"✓ True Negatives: {tn:,}")
        print(f"✓ False Positives: {fp:,}")
        print(f"✓ False Negatives: {fn:,}")
        print(f"✓ True Positives: {tp:,}")
        metrics_out["confusion_matrix_breakdown"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # --- 8. Save model and label map ---
    os.makedirs('models', exist_ok=True)
    model_path = f"models/genesis_{'multiclass' if is_multiclass else 'binary'}.pth"
    torch.save(model.state_dict(), model_path)
    label_map = {0: 'Other', 1: 'CHEK2', 2: 'TBX1', 3: 'COMT'}
    with open('models/label_map.json', 'w') as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    # Save metrics and the text report
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics_out, f, indent=2)
    with open('models/classification_report.txt', 'w') as f:
        f.write(report_text)
    print(f"\n✓ Saved model to {model_path}, label map and metrics to models/")

if __name__ == "__main__":
    main()