# config.py — Cấu hình standalone cho OTA_KG Vietnamese (PhoBERT).
# File này cho phép chạy ontology/owl_to_neo4j.py và data/dataset.py
# độc lập ngoài notebook.
#
# Credentials được đọc từ biến môi trường / file .env (KHÔNG hard-code).
# Tạo file .env từ .env.example rồi điền thông tin thực.

from __future__ import annotations
import os
import torch
from pathlib import Path

# ── Hỗ trợ .env qua python-dotenv (optional) ─────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv chưa cài — đọc từ môi trường hệ thống

# ── Phát hiện môi trường ──────────────────────────────────────────────────────
IS_KAGGLE = os.path.exists("/kaggle/input")

if IS_KAGGLE:
    _candidates = [
        Path("/kaggle/input/ota-kg"),
        Path("/kaggle/input/datasets/nguyentam306/ota-kg"),
    ]
    KAGGLE_INPUT_DIR = _candidates[0]
    for _c in _candidates:
        if _c.exists():
            KAGGLE_INPUT_DIR = _c
            break

    INPUT_DATA_DIR = KAGGLE_INPUT_DIR / "data"
    ONTOLOGY_DIR   = KAGGLE_INPUT_DIR / "ontology"

    BASE_DIR       = Path.cwd()
    DATA_DIR       = BASE_DIR / "data"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"

    INTENT_FILE   = INPUT_DATA_DIR / "intent" / "TranfomerVN.json"
    POSITIVE_FILE = INPUT_DATA_DIR / "positive_vi.txt"
    NEGATIVE_FILE = INPUT_DATA_DIR / "negative_vi.txt"
    NEGATION_FILE = INPUT_DATA_DIR / "negation_words_vi.csv"

    KG_CACHE_FILE = DATA_DIR / "kg_cache.json"
    if not KG_CACHE_FILE.exists() and (INPUT_DATA_DIR / "kg_cache.json").exists():
        KG_CACHE_FILE = INPUT_DATA_DIR / "kg_cache.json"
else:
    BASE_DIR       = Path(__file__).parent
    DATA_DIR       = BASE_DIR / "data"
    ONTOLOGY_DIR   = BASE_DIR / "ontology"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"

    INTENT_FILE   = DATA_DIR / "intent" / "TranfomerVN.json"
    POSITIVE_FILE = DATA_DIR / "positive_vi.txt"
    NEGATIVE_FILE = DATA_DIR / "negative_vi.txt"
    NEGATION_FILE = DATA_DIR / "negation_words_vi.csv"
    KG_CACHE_FILE = DATA_DIR / "kg_cache.json"

CHECKPOINT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── TestComputer ──────────────────────────────────────────────────────────────
def _find_dir(base: Path, data_dir: Path, name: str) -> Path:
    _kaggle_inp = KAGGLE_INPUT_DIR if IS_KAGGLE else None
    _cands = [
        (_kaggle_inp / name) if _kaggle_inp else None,
        data_dir.parent / name,
        data_dir / name,
        base / name,
        base.parent / name,
    ]
    for _c in _cands:
        if _c and _c.exists():
            return _c
    return base / name

TEST_DIR          = _find_dir(BASE_DIR, DATA_DIR, "TestComputer")
REAL_VI_DIR       = TEST_DIR / "Real_VI"
TESTING_ACC_FILE  = REAL_VI_DIR / "Testing_VI.txt"
TESTING_ACC2_FILE = REAL_VI_DIR / "ReadAcc_VI.txt"

if not TESTING_ACC_FILE.exists():
    _alt = REAL_VI_DIR / "Testing_ACC.txt"
    if _alt.exists():
        TESTING_ACC_FILE = _alt

# ── OWL Ontology — BUG FIX: bọc tên file trong list ─────────────────────────
# TRƯỚC (sai): [ONTOLOGY_DIR / f for f in "N_P_Ontology_VI_v2.owl"]
#   → lặp qua từng ký tự, tạo ~20 path rác
# SAU  (đúng): [ONTOLOGY_DIR / f for f in ["N_P_Ontology_VI_v2.owl"]]
OWL_FILES = [ONTOLOGY_DIR / f for f in ["N_P_Ontology_VI_v2.owl"]]
OWL_FILE  = OWL_FILES[0]  # alias cho trường hợp chỉ dùng 1 file

# ── Labels ────────────────────────────────────────────────────────────────────
LABEL2ID = {"N1": 0, "N2": 1, "P1": 2, "P2": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = 4

# ── Tag mapping TranfomerVN.json → label ─────────────────────────────────────
OLD_TAG_TO_NEW = {
    "Tiêu cực loại 1": "N1", "Tiêu cực loại 2": "N2",
    "Tích cực loại 1": "P1", "Tích cực loại 2": "P2",
    "Tiêu cực Loại 1": "N1", "Tiêu cực Loại 2": "N2",
    "Tích cực Loại 1": "P1", "Tích cực Loại 2": "P2",
    "negative-1": "N1",      "negative-2": "N2",
    "positive-1": "P1",      "positive-2": "P2",
}

# ── Neo4j — đọc từ biến môi trường hoặc .env ─────────────────────────────────
# KHÔNG hard-code credentials ở đây. Tạo file .env từ .env.example.
NEO4J_URI      = os.getenv("NEO4J_URI",      "neo4j+s://a37377cb.databases.neo4j.io")
NEO4J_USER     = os.getenv("NEO4J_USER",     "a37377cb")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "34dFsbuQOue822_fhjBlhZABUZSXNGuIpzcLaXGFRuw")

if not NEO4J_PASSWORD:
    import warnings
    warnings.warn(
        "[Config] NEO4J_PASSWORD chưa được set. "
        "Tạo file .env từ .env.example và điền mật khẩu Neo4j Aura.",
        stacklevel=2,
    )

# ── PhoBERT ───────────────────────────────────────────────────────────────────
BERT_MODEL_NAME    = "vinai/phobert-base"
BERT_MAX_LENGTH    = 128
BERT_HIDDEN_DIM    = 768
BERT_FREEZE_LAYERS = 2
BERT_POOLING       = "mean_cls"

# ── KG ───────────────────────────────────────────────────────────────────────
KG_DIM = 64

# ── Fusion ────────────────────────────────────────────────────────────────────
FUSION_TYPE    = "mlp"
MLP_HIDDEN_DIM = 384
MLP_DROPOUT    = 0.4

# ── OTA ───────────────────────────────────────────────────────────────────────
OTA_ALPHA           = 0.5
OTA_LEARNABLE_ALPHA = True

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE              = 16
NUM_EPOCHS              = 30
LR_BERT                 = 2e-5
LR_REST                 = 5e-4
WEIGHT_DECAY            = 0.01
TRAIN_VAL_SPLIT         = 0.8
RANDOM_SEED             = 42
EARLY_STOPPING_PATIENCE = 7
LABEL_SMOOTHING         = 0.1
DOMAIN_DATA_MULTIPLIER  = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"[Config VN] BASE_DIR      : {BASE_DIR}")
    print(f"[Config VN] ONTOLOGY_DIR  : {ONTOLOGY_DIR}")
    print(f"[Config VN] OWL_FILES     : {OWL_FILES}")
    print(f"[Config VN] OWL_FILE exists: {OWL_FILE.exists()}")
    print(f"[Config VN] BERT Model    : {BERT_MODEL_NAME}")
    print(f"[Config VN] Device        : {DEVICE}")
    print(f"[Config VN] NEO4J_URI     : {NEO4J_URI}")
    print(f"[Config VN] NEO4J_USER    : {NEO4J_USER}")
    print(f"[Config VN] Password set  : {'YES' if NEO4J_PASSWORD else 'NO — set NEO4J_PASSWORD in .env'}")
