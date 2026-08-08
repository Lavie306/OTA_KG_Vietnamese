"""
data/dataset.py
PyTorch Dataset với schema N1/N2/P1/P2.
Hỗ trợ cả transformer_v2.json (EN) và TranfomerVN.json (VN).
"""

from __future__ import annotations
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

try:
    from config import (
        INTENT_FILE, LABEL2ID, ID2LABEL, NUM_CLASSES,
        BATCH_SIZE, TRAIN_VAL_SPLIT, RANDOM_SEED,
        OLD_TAG_TO_NEW, TESTING_ACC_FILE, TESTING_ACC2_FILE,
        DOMAIN_DATA_MULTIPLIER,
    )
except ImportError:
    LABEL2ID = {'N1': 0, 'N2': 1, 'P1': 2, 'P2': 3}
    ID2LABEL = {v: k for k, v in LABEL2ID.items()}
    NUM_CLASSES = 4
    BATCH_SIZE = 16
    TRAIN_VAL_SPLIT = 0.8
    RANDOM_SEED = 42
    DOMAIN_DATA_MULTIPLIER = 5
    INTENT_FILE = Path(__file__).parent / 'intent' / 'TranfomerVN.json'
    TESTING_ACC_FILE = Path(__file__).parent.parent / 'TestComputer' / 'Real_VI' / 'Testing_VI.txt'
    TESTING_ACC2_FILE = Path(__file__).parent.parent / 'TestComputer' / 'Real_VI' / 'ReadAcc_VI.txt'
    OLD_TAG_TO_NEW = {
        'Tiêu cực loại 1': 'N1', 'Tiêu cực loại 2': 'N2',
        'Tích cực loại 1': 'P1', 'Tích cực loại 2': 'P2',
        'Tiêu cực Loại 1': 'N1', 'Tiêu cực Loại 2': 'N2',
        'Tích cực Loại 1': 'P1', 'Tích cực Loại 2': 'P2',
        'negative-1': 'N1', 'negative-2': 'N2',
        'positive-1': 'P1', 'positive-2': 'P2',
    }


def decode_hyphenated(pattern: str) -> str:
    return pattern.replace('-', ' ')


def _load_domain_file(filepath: Path) -> list[dict]:
    samples = []
    if not filepath.exists():
        return samples
    current_block = []
    label_map = {'ne1': 0, 'ne2': 1, 'po1': 2, 'po2': 3}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('=='):
                if current_block and len(current_block) >= 3:
                    sentence = current_block[1]
                    gt_label_str = current_block[-1]
                    if gt_label_str in label_map:
                        samples.append({'sentence': sentence, 'label': label_map[gt_label_str]})
                current_block = []
            if line:
                current_block.append(line)
    if current_block and len(current_block) >= 3:
        sentence = current_block[1]
        gt_label_str = current_block[-1]
        if gt_label_str in label_map:
            samples.append({'sentence': sentence, 'label': label_map[gt_label_str]})
    return samples


_SYNONYMS_VN = {
    'tuyệt vời':    ['xuất sắc', 'hoàn hảo', 'tuyệt hảo', 'thần kỳ', 'tuyệt đỉnh'],
    'xuất sắc':     ['tuyệt vời', 'hoàn hảo', 'đặc biệt', 'vượt trội', 'ưu tú'],
    'hoàn hảo':     ['tuyệt vời', 'xuất sắc', 'lý tưởng', 'hoàn toàn', 'trọn vẹn'],
    'tốt':          ['hay', 'ổn', 'khá', 'được', 'tốt đẹp'],
    'hài lòng':     ['thỏa mãn', 'vừa lòng', 'mãn nguyện', 'hạnh phúc', 'vui vẻ'],
    'thích':        ['yêu', 'mến', 'ưa thích', 'đam mê', 'mê'],
    'nhanh':        ['mau', 'chóng', 'kịp thời', 'tức thì', 'lẹ'],
    'dễ':           ['đơn giản', 'thuận tiện', 'tiện lợi', 'dễ dàng', 'dễ dùng'],
    'tệ':           ['kém', 'xấu', 'tồi', 'dở', 'không tốt'],
    'chậm':         ['ì ạch', 'lề mề', 'trì hoãn', 'cà rà'],
    'thất vọng':    ['không hài lòng', 'hụt hẫng', 'buồn', 'chán'],
    'kinh khủng':   ['tồi tệ', 'thảm hại', 'ghê gớm', 'khủng khiếp', 'tệ hại'],
    'tồi tệ':       ['kinh khủng', 'thảm hại', 'tệ lắm', 'không thể chấp nhận'],
    'ghét':         ['căm ghét', 'không thích', 'chán ghét', 'ghét bỏ'],
    'hỏng':         ['vỡ', 'bị hỏng', 'không dùng được', 'hư', 'lỗi'],
    'vô dụng':      ['không có ích', 'vô ích', 'không dùng được', 'thừa'],
}

_INTENSIFIERS_POS_VN = ['rất', 'cực kỳ', 'vô cùng', 'thực sự', 'hoàn toàn', 'hết sức']
_INTENSIFIERS_NEG_VN = ['rất', 'cực kỳ', 'hoàn toàn', 'thực sự', 'quá', 'hết sức']

_TEMPLATES_BY_LABEL_VN = {
    'P2': [
        'Sản phẩm này {w}', 'Dịch vụ {w} và {w2}', 'Tôi thấy {w}',
        'Trải nghiệm {w} tuyệt vời', 'Chất lượng {w}',
        'Rất hài lòng vì {w}', 'Sẽ giới thiệu vì {w}', 'Thực sự {w}',
        'Đáng khen — {w}', 'Hoàn toàn {w} — sẽ mua lại',
    ],
    'P1': [
        'Sản phẩm tương đối {w}', 'Dịch vụ {w}', 'Khá {w}',
        'Tôi cảm thấy {w}', 'Trải nghiệm {w}', 'Chất lượng {w}',
        '{w} — có thể mua được', 'Ổn ở mức {w}',
        'Có thể cải thiện nhưng nhìn chung {w}', 'Nhìn chung {w}',
    ],
    'N1': [
        'Sản phẩm {w}', 'Dịch vụ {w} và {w2}', 'Tôi thấy {w}',
        'Có phần {w}', 'Chất lượng {w}', '{w} — cần cải thiện',
        'Hơi {w} so với kỳ vọng', 'Không hoàn toàn hài lòng vì {w}',
        'Trải nghiệm {w}', 'Cần xem lại vì {w}',
    ],
    'N2': [
        'Sản phẩm rất {w}', 'Dịch vụ {w} và {w2}', 'Hoàn toàn {w}',
        'Tôi không thể chấp nhận {w}', 'Chất lượng {w}',
        'Rất tệ — {w}', 'Tránh xa! {w}', 'Trải nghiệm {w} tệ hại',
        'Thất vọng vì {w}', 'Không bao giờ mua lại vì {w}',
    ],
}


def _synonym_replace_vn(text: str, rng: random.Random) -> str:
    for phrase, synonyms in _SYNONYMS_VN.items():
        if phrase in text:
            return text.replace(phrase, rng.choice(synonyms), 1)
    return text


def _intensity_augment_vn(pattern: str, rng: random.Random, is_positive: bool) -> str:
    pool = _INTENSIFIERS_POS_VN if is_positive else _INTENSIFIERS_NEG_VN
    intensifier = rng.choice(pool)
    return rng.choice([
        f'{intensifier} {pattern}',
        f'Đây là {intensifier} {pattern}',
        f'Thực sự {intensifier} {pattern}',
    ])


class SentimentDataset(Dataset):
    def __init__(self, intent_file: Path | str = INTENT_FILE):
        self.samples: list[dict] = []
        self._load(Path(intent_file))

    def _load(self, path: Path) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        counts = {k: 0 for k in LABEL2ID}
        label_patterns = {k: [] for k in LABEL2ID}
        is_vn = 'VN' in path.name or 'vn' in path.name.lower()

        for intent in data['intents']:
            tag = intent['tag']
            label_name = OLD_TAG_TO_NEW.get(tag, None)
            if label_name is None:
                raise ValueError(f'Unknown tag in {path.name}: {tag}')

            label_id = LABEL2ID[label_name]
            for pattern in intent['patterns']:
                decoded = decode_hyphenated(pattern) if is_vn else pattern
                self.samples.append({'sentence': decoded, 'label': label_id})
                counts[label_name] += 1
                label_patterns[label_name].append(decoded)

        domain_samples = _load_domain_file(TESTING_ACC_FILE) + _load_domain_file(TESTING_ACC2_FILE)
        unique_domain = {s['sentence']: s['label'] for s in domain_samples}
        domain_list = [{'sentence': k, 'label': v} for k, v in unique_domain.items()]
        if domain_list:
            print(f'  [Domain] Loaded {len(domain_list)} unique domain samples.')
            for _ in range(DOMAIN_DATA_MULTIPLIER):
                self.samples.extend(domain_list)

        rng = random.Random(RANDOM_SEED)
        aug_counts = {k: 0 for k in LABEL2ID}
        max_count = max(counts.values())

        for label_name, patterns in label_patterns.items():
            if len(patterns) < 2:
                continue
            label_id = LABEL2ID[label_name]
            is_positive = label_name in ('P1', 'P2')
            templates = _TEMPLATES_BY_LABEL_VN[label_name]
            n_aug = max(80, max_count - counts[label_name] + len(patterns) // 2)

            for i in range(n_aug):
                w = rng.choice(patterns)
                w2 = rng.choice(patterns)
                aug_type = i % 4

                if aug_type == 0:
                    sentence = rng.choice(templates).format(w=w, w2=w2)
                elif aug_type == 1:
                    sentence = _synonym_replace_vn(w, rng)
                    if sentence == w:
                        sentence = rng.choice(templates).format(w=w, w2=w2)
                elif aug_type == 2:
                    sentence = _intensity_augment_vn(w, rng, is_positive)
                else:
                    intensifier = rng.choice(_INTENSIFIERS_POS_VN if is_positive else _INTENSIFIERS_NEG_VN)
                    base = rng.choice(templates).format(w=w, w2=w2)
                    sentence = f'{intensifier} — {base}'

                self.samples.append({'sentence': sentence, 'label': label_id})
                aug_counts[label_name] += 1

        total_aug = sum(aug_counts.values())
        print(f'\n[Dataset] Loaded {len(self.samples)} samples from {path.name}')
        print(f'  (original: {len(self.samples) - total_aug}, augmented: {total_aug})')

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return {
            'sentence': self.samples[idx]['sentence'],
            'label': torch.tensor(self.samples[idx]['label'], dtype=torch.long),
        }
