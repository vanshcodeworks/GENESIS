import json
import os
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool  # NEW

MODEL_PATH = os.getenv('GENESIS_MODEL_PATH', os.path.join('models', 'genesis_multiclass.pth'))
LABELS_PATH = os.getenv('GENESIS_LABELS_PATH', os.path.join('models', 'label_map.json'))

# Limit CPU threads to avoid stalls on Windows
torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2))))  # NEW

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

# --- Lazy-loading artifacts instead of hard-failing at import ---
LABEL_MAP = None
NUM_CLASSES = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = None
_LAST_LOAD_ERROR = None

def _load_artifacts():
    global LABEL_MAP, NUM_CLASSES, MODEL, _LAST_LOAD_ERROR
    if not os.path.exists(LABELS_PATH) or not os.path.exists(MODEL_PATH):
        _LAST_LOAD_ERROR = (
            "Model artifacts not found. Train first: python scripts/02_train_model.py. "
            f"Expected files:\n - {LABELS_PATH}\n - {MODEL_PATH}"
        )
        return False
    try:
        with open(LABELS_PATH, 'r') as f:
            LABEL_MAP = {int(k): v for k, v in json.load(f).items()}
        NUM_CLASSES = max(LABEL_MAP.keys()) + 1
        model = DNA_CNN_MultiClass(num_classes=NUM_CLASSES).to(DEVICE)
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(state, dict) and any(isinstance(k, str) and k.startswith('net.') for k in state.keys()):
            state = {k.replace('net.', 'conv_net.'): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        MODEL = model
        _LAST_LOAD_ERROR = None
        return True
    except Exception as e:
        _LAST_LOAD_ERROR = f"Failed to load model/labels: {e}"
        MODEL = None
        LABEL_MAP = None
        NUM_CLASSES = None
        return False

app = FastAPI(title='GENESIS API', version='1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    _load_artifacts()

@app.get('/health')  # NEW
async def health():
    return {"ok": True, "has_model": MODEL is not None}

class PredictIn(BaseModel):
    sequence: str  # 200bp

def _infer_tensor(X_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = MODEL(X_tensor)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]

@app.post('/predict')
async def predict(inp: PredictIn):
    if MODEL is None or LABEL_MAP is None:
        # Try loading on-demand (e.g., artifacts added after startup)
        if not _load_artifacts():
            raise HTTPException(status_code=503, detail=_LAST_LOAD_ERROR or "Model not available")
    seq = inp.sequence
    if len(seq) != 200:
        raise HTTPException(status_code=400, detail='sequence must be 200bp')
    X = one_hot(seq)
    X = np.transpose(X, (1, 0))[None, ...]
    X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    try:
        probs = await run_in_threadpool(_infer_tensor, X)  # NEW: avoid blocking event loop
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")
    pred_idx = int(np.argmax(probs))
    return {
        'prediction_index': pred_idx,
        'prediction_label': LABEL_MAP.get(pred_idx, str(pred_idx)),
        'probabilities': {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(probs)}
    }

@app.get('/')
async def root():
    return {
        "status": "ok",
        "model": "GENESIS",
        "device": DEVICE,
        "has_model": MODEL is not None,
        "expected_paths": {"labels": LABELS_PATH, "model": MODEL_PATH},
        "last_error": _LAST_LOAD_ERROR,
    }
