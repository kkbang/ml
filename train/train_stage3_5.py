"""
train_stage5.py — Stage 4b 위에 structural_twin hard negative 강화 학습

핵심:
  - 새 hard negative (FP에서 추출한 119개 structural_twin) 사용
  - hard_neg_ratio 0.3 → 0.5 (배치 절반이 hard neg)
  - per-sample weight: fp_similarity 기반 (헷갈렸을수록 강한 push)
  - LR 1e-6 → 3e-7 (collapse 방지)
  - max_epochs 30, patience 5
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/shinmk/code-killr"))

sys.path.insert(0, f"{PROJECT_ROOT}/core")
sys.path.append(f"{PROJECT_ROOT}/parser")

import torch
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from model import GraphCodeBERTEncoder
from loss import weighted_nt_xent_loss
from dataset import PairDataset, encode_with_dfg, build_attn_mask, TOTAL_LENGTH
from tqdm import tqdm

CONFIG = {
    'train_path':    str(PROJECT_ROOT / "data/train_v3_anon.jsonl"),
    'hard_neg_path': str(PROJECT_ROOT / "data/hard_negatives_combined.jsonl"),
    'val_path':      str(PROJECT_ROOT / "data/val_v3_anon.jsonl"),
    'model_path':    str(PROJECT_ROOT / "model/GCB_dfg_stage3.pt"),
    'save_path':     str(PROJECT_ROOT / "model/GCB_dfg_stage3_5.pt"),
    'log_path':      'train_gcb_stage3_5_log.txt',
    'model_name':    'microsoft/graphcodebert-base',
    'batch_size':    32,
    'lr':            1e-6,               # 강한 push로 인한 collapse 방지
    'weight_decay':  0.01,
    'temperature':   0.07,
    'max_epochs':    30,
    'patience':      5,
    'warmup_ratio':  0.05,                # 짧은 warmup
    'hard_neg_ratio': 0.3,                # 30% → 50%로 강화
    'aug_prob':      0.3,
    'aug_ratio_min': 0.25,
    'aug_ratio_max': 1.0,
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
}

LICENSE_WEIGHTS = {
    'GPL-2.0': 3.0, 'GPL-3.0': 3.0, 'AGPL-3.0': 3.0,
    'LGPL-2.1': 2.0, 'LGPL-3.0': 2.0,
}
HARD_NEG_BASE_WEIGHT = 2.5  # 기본 weight 2.5 → 3.0


def compute_hard_neg_weight(fp_sim: float) -> float:
    """
    fp_similarity 기반 동적 weighting.
    모델이 헷갈렸을수록 강한 gradient.

    sim 0.40 → weight 3.0
    sim 0.50 → weight 4.0
    sim 0.60 → weight 5.0
    sim 0.70 → weight 6.0
    sim 0.79 → weight 6.9 (max in our data)
    """
    return HARD_NEG_BASE_WEIGHT + max(0.0, fp_sim - 0.4) * 5.0


class Stage5Dataset(Dataset):
    def __init__(self, train_path, hard_neg_path, tokenizer,
                 hard_neg_ratio=0.5, aug_prob=0.3,
                 aug_ratio_min=0.25, aug_ratio_max=1.0):
        self.tokenizer      = tokenizer
        self.hard_neg_ratio = hard_neg_ratio
        self.aug_prob       = aug_prob
        self.aug_ratio_min  = aug_ratio_min
        self.aug_ratio_max  = aug_ratio_max

        with open(train_path) as f:
            self.positives = [json.loads(l) for l in f]
        with open(hard_neg_path) as f:
            self.hard_negatives = [json.loads(l) for l in f]

        # hard negative weight 미리 계산
        weights = [compute_hard_neg_weight(hn.get('fp_similarity', 0.4))
                   for hn in self.hard_negatives]
        print(f"Positive pair: {len(self.positives)}개")
        print(f"Hard Negative: {len(self.hard_negatives)}개")
        print(f"  weight 분포: min={min(weights):.2f}, max={max(weights):.2f}, "
              f"mean={sum(weights)/len(weights):.2f}")

    def _encode(self, code, language):
        ids, pos, d2c, d2d = encode_with_dfg(code, language, self.tokenizer)
        mask = build_attn_mask(pos, d2c, d2d)
        return (
            torch.tensor(ids,  dtype=torch.long),
            torch.tensor(pos,  dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )

    def _make_item(self, anchor_code, pair_code, language, weight):
        a_ids, a_pos, a_mask = self._encode(anchor_code, language)
        p_ids, p_pos, p_mask = self._encode(pair_code,   language)
        return {
            'anchor_input_ids':      a_ids,
            'anchor_position_ids':   a_pos,
            'anchor_attn_mask':      a_mask,
            'positive_input_ids':    p_ids,
            'positive_position_ids': p_pos,
            'positive_attn_mask':    p_mask,
            'weight': torch.tensor(weight, dtype=torch.float),
        }

    def __len__(self):
        return len(self.positives)

    def __getitem__(self, idx):
        # 강화된 hard_neg_ratio = 0.5: 절반은 hard negative
        if self.hard_negatives and np.random.random() < self.hard_neg_ratio:
            hn = self.hard_negatives[np.random.randint(len(self.hard_negatives))]
            lang = hn.get('language', 'python')
            try:
                # fp_similarity 기반 동적 weight
                weight = compute_hard_neg_weight(hn.get('fp_similarity', 0.4))
                # 라이선스 가중치도 고려
                lic_weight = LICENSE_WEIGHTS.get(hn.get('anchor_license', ''), 1.0)
                final_weight = weight * lic_weight if lic_weight > 1.0 else weight
                return self._make_item(hn['anchor'], hn['negative'], lang, final_weight)
            except Exception:
                pass

        pair = self.positives[idx]
        try:
            lang   = pair.get('language', 'python')
            weight = LICENSE_WEIGHTS.get(pair.get('license', ''), 1.0)
            return self._make_item(pair['anchor'], pair['positive'], lang, weight)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.positives))


def compute_fpr(model, val_loader, device, threshold=0.7) -> float:
    """threshold 0.7로 변경 (운영 기준)."""
    model.eval()
    fp_count = total_neg = 0
    with torch.no_grad():
        for batch in val_loader:
            a_emb = F.normalize(model(
                batch['anchor_input_ids'].to(device),
                batch['anchor_position_ids'].to(device),
                batch['anchor_attn_mask'].to(device),
            ), dim=1)
            p_emb = F.normalize(model(
                batch['positive_input_ids'].to(device),
                batch['positive_position_ids'].to(device),
                batch['positive_attn_mask'].to(device),
            ), dim=1)
            B = a_emb.shape[0]
            if B < 2:
                continue
            sim  = torch.matmul(a_emb, p_emb.T)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)
            neg  = sim[mask]
            fp_count  += (neg > threshold).sum().item()
            total_neg += neg.numel()
    return fp_count / total_neg if total_neg > 0 else 0.0


def train_stage5():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print(f"Device: {CONFIG['device']}")
    print(f"Hard neg ratio: {CONFIG['hard_neg_ratio']}")
    print(f"LR: {CONFIG['lr']}\n")

    train_set = Stage5Dataset(
        CONFIG['train_path'], CONFIG['hard_neg_path'], tokenizer,
        hard_neg_ratio=CONFIG['hard_neg_ratio'],
        aug_prob=CONFIG['aug_prob'],
        aug_ratio_min=CONFIG['aug_ratio_min'],
        aug_ratio_max=CONFIG['aug_ratio_max'],
    )
    val_set = PairDataset(CONFIG['val_path'], tokenizer)

    train_loader = DataLoader(train_set, batch_size=CONFIG['batch_size'],
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=CONFIG['batch_size'],
                              shuffle=False, num_workers=4, pin_memory=True)

    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    ckpt    = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    model.load_state_dict(ckpt.get('model', ckpt))
    model.encoder.gradient_checkpointing_enable()
    model = model.to(CONFIG['device'])
    print(f"Stage 4b 모델 로드: {CONFIG['model_path']}\n")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])
    total_steps  = len(train_loader) * CONFIG['max_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler()

    best_val_loss    = float('inf')
    best_fpr         = float('inf')
    patience_counter = 0

    with open(CONFIG['log_path'], 'w') as log:
        log.write("epoch,train_loss,val_loss,fpr@0.7\n")

        for epoch in range(1, CONFIG['max_epochs'] + 1):
            model.train()
            train_loss = 0.0
            bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
            for batch in bar:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    a_emb = model(batch['anchor_input_ids'].to(CONFIG['device']),
                                  batch['anchor_position_ids'].to(CONFIG['device']),
                                  batch['anchor_attn_mask'].to(CONFIG['device']))
                    p_emb = model(batch['positive_input_ids'].to(CONFIG['device']),
                                  batch['positive_position_ids'].to(CONFIG['device']),
                                  batch['positive_attn_mask'].to(CONFIG['device']))
                    loss = weighted_nt_xent_loss(a_emb, p_emb,
                                                 batch['weight'].to(CONFIG['device']),
                                                 CONFIG['temperature'])
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()
                bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        a_emb = model(batch['anchor_input_ids'].to(CONFIG['device']),
                                      batch['anchor_position_ids'].to(CONFIG['device']),
                                      batch['anchor_attn_mask'].to(CONFIG['device']))
                        p_emb = model(batch['positive_input_ids'].to(CONFIG['device']),
                                      batch['positive_position_ids'].to(CONFIG['device']),
                                      batch['positive_attn_mask'].to(CONFIG['device']))
                        loss = weighted_nt_xent_loss(a_emb, p_emb,
                                                     batch['weight'].to(CONFIG['device']),
                                                     CONFIG['temperature'])
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            fpr     = compute_fpr(model, val_loader, CONFIG['device'], threshold=0.7)
            print(f"[Epoch {epoch}] train: {avg_train:.4f} | val: {avg_val:.4f} | fpr@0.7: {fpr:.4f}")
            log.write(f"{epoch},{avg_train:.4f},{avg_val:.4f},{fpr:.4f}\n")
            log.flush()

            # 저장 기준: val_loss 우선, fpr 기록
            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                best_fpr         = fpr
                patience_counter = 0
                torch.save({'epoch': epoch, 'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                            'best_val_loss': best_val_loss, 'fpr': fpr}, CONFIG['save_path'])
                print(f"  → best 저장 (val: {avg_val:.4f}, fpr@0.7: {fpr:.4f})")
            else:
                patience_counter += 1
                print(f"  → patience {patience_counter}/{CONFIG['patience']}")
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping! (epoch {epoch})")
                    break

    print(f"\nStage 5 완료. best val loss: {best_val_loss:.4f} | fpr@0.7: {best_fpr:.4f}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_stage5()