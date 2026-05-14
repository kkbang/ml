# /home/ngseokim/code-killr/train_stage3.py
"""
Stage 3: Rename Augmentation Fine-tuning
목적: DFG가 식별자 이름이 아닌 구조적 패턴에 의존하도록 강제
방법: positive를 on-the-fly로 현실적 rename해서 학습
"""
import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')
sys.path.append('/home/ngseokim/code-killr/parser')

import torch
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from model import GraphCodeBERTEncoder
from loss import weighted_nt_xent_loss
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')
sys.path.append('/home/ngseokim/code-killr/parser')

from dataset import PairDataset, encode_with_dfg, build_attn_mask, TOTAL_LENGTH
from augment import realistic_rename_identifiers

CONFIG = {
    'train_path':    '/home/ngseokim/code-killr/data/train_v3.jsonl',
    'hard_neg_path': '/home/ngseokim/code-killr/data/hard_negatives_gcb.jsonl',
    'val_path':      '/home/ngseokim/code-killr/data/val_v3.jsonl',
    'model_path':    '/home/ngseokim/code-killr/model/GCB_dfg_stage2.pt',
    'save_path':     '/home/ngseokim/code-killr/model/GCB_dfg_stage3.pt',
    'log_path':      'train_gcb_stage3_log.txt',
    'model_name':    'microsoft/graphcodebert-base',
    'batch_size':    32,
    'lr':            5e-6,        # Stage2(1e-5)의 절반
    'weight_decay':  0.01,
    'temperature':   0.07,
    'max_epochs':    15,          # Stage3는 짧게
    'patience':      5,
    'warmup_ratio':  0.1,
    'hard_neg_ratio': 0.3,
    'aug_prob':      0.7,         # 70% 확률로 positive rename 적용
    'aug_ratio_min': 0.25,        # rename 비율 최소
    'aug_ratio_max': 0.75,        # rename 비율 최대 (100%는 너무 극단적)
    'device':        'cuda' if torch.cuda.is_available() else 'cpu',
}

LICENSE_WEIGHTS = {
    'GPL-2.0': 3.0, 'GPL-3.0': 3.0, 'AGPL-3.0': 3.0,
    'LGPL-2.1': 2.0, 'LGPL-3.0': 2.0,
}
HARD_NEG_WEIGHT = 2.5


class Stage3Dataset(Dataset):
    """
    Stage2Dataset + positive rename augmentation.
    aug_prob 확률로 positive의 변수명을 현실적으로 rename.
    """
    def __init__(self, train_path, hard_neg_path, tokenizer,
                 hard_neg_ratio=0.3, aug_prob=0.7,
                 aug_ratio_min=0.25, aug_ratio_max=0.75):
        self.tokenizer      = tokenizer
        self.hard_neg_ratio = hard_neg_ratio
        self.aug_prob       = aug_prob
        self.aug_ratio_min  = aug_ratio_min
        self.aug_ratio_max  = aug_ratio_max

        with open(train_path) as f:
            self.positives = [json.loads(l) for l in f]
        with open(hard_neg_path) as f:
            self.hard_negatives = [json.loads(l) for l in f]

        print(f"Positive pair: {len(self.positives)}개")
        print(f"Hard Negative: {len(self.hard_negatives)}개")
        print(f"Augmentation: prob={aug_prob}, ratio=[{aug_ratio_min},{aug_ratio_max}]")

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
        # Hard Negative 샘플 (augmentation 없이)
        if self.hard_negatives and np.random.random() < self.hard_neg_ratio:
            hn = self.hard_negatives[np.random.randint(len(self.hard_negatives))]
            lang = hn.get('language', 'python')
            try:
                weight = LICENSE_WEIGHTS.get(hn.get('anchor_license', ''), HARD_NEG_WEIGHT)
                return self._make_item(hn['anchor'], hn['negative'], lang, weight)
            except Exception:
                pass

        # Positive pair + rename augmentation
        pair = self.positives[idx]
        try:
            lang     = pair.get('language', 'python')
            weight   = LICENSE_WEIGHTS.get(pair.get('license', ''), 1.0)
            positive = pair['positive']

            # aug_prob 확률로 positive rename
            if np.random.random() < self.aug_prob:
                ratio    = np.random.uniform(self.aug_ratio_min, self.aug_ratio_max)
                positive = realistic_rename_identifiers(positive, lang, ratio)

            return self._make_item(pair['anchor'], positive, lang, weight)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.positives))


def compute_fpr(model, val_loader, device, threshold=0.5) -> float:
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


def train_stage3():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print(f"Device: {CONFIG['device']}")

    train_set = Stage3Dataset(
        CONFIG['train_path'], CONFIG['hard_neg_path'], tokenizer,
        hard_neg_ratio=CONFIG['hard_neg_ratio'],
        aug_prob=CONFIG['aug_prob'],
        aug_ratio_min=CONFIG['aug_ratio_min'],
        aug_ratio_max=CONFIG['aug_ratio_max'],
    )
    val_set = PairDataset(CONFIG['val_path'], tokenizer)

    train_loader = DataLoader(
        train_set, batch_size=CONFIG['batch_size'],
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    ckpt    = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
    model.load_state_dict(ckpt.get('model', ckpt))
    model.encoder.gradient_checkpointing_enable()
    model = model.to(CONFIG['device'])
    print(f"Stage2 모델 로드 완료: {CONFIG['model_path']}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'],
    )
    total_steps  = len(train_loader) * CONFIG['max_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler()

    best_val_loss    = float('inf')
    patience_counter = 0

    with open(CONFIG['log_path'], 'w') as log:
        log.write("epoch,train_loss,val_loss,fpr\n")

        for epoch in range(1, CONFIG['max_epochs'] + 1):
            model.train()
            train_loss = 0.0
            bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

            for batch in bar:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    a_emb = model(
                        batch['anchor_input_ids'].to(CONFIG['device']),
                        batch['anchor_position_ids'].to(CONFIG['device']),
                        batch['anchor_attn_mask'].to(CONFIG['device']),
                    )
                    p_emb = model(
                        batch['positive_input_ids'].to(CONFIG['device']),
                        batch['positive_position_ids'].to(CONFIG['device']),
                        batch['positive_attn_mask'].to(CONFIG['device']),
                    )
                    loss = weighted_nt_xent_loss(
                        a_emb, p_emb,
                        batch['weight'].to(CONFIG['device']),
                        CONFIG['temperature'],
                    )

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
                        a_emb = model(
                            batch['anchor_input_ids'].to(CONFIG['device']),
                            batch['anchor_position_ids'].to(CONFIG['device']),
                            batch['anchor_attn_mask'].to(CONFIG['device']),
                        )
                        p_emb = model(
                            batch['positive_input_ids'].to(CONFIG['device']),
                            batch['positive_position_ids'].to(CONFIG['device']),
                            batch['positive_attn_mask'].to(CONFIG['device']),
                        )
                        loss = weighted_nt_xent_loss(
                            a_emb, p_emb,
                            batch['weight'].to(CONFIG['device']),
                            CONFIG['temperature'],
                        )
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            fpr     = compute_fpr(model, val_loader, CONFIG['device'])

            print(f"[Epoch {epoch}] train: {avg_train:.4f} | val: {avg_val:.4f} | fpr: {fpr:.4f}")
            log.write(f"{epoch},{avg_train:.4f},{avg_val:.4f},{fpr:.4f}\n")
            log.flush()

            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                patience_counter = 0
                torch.save({
                    'epoch':         epoch,
                    'model':         model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'scheduler':     scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, CONFIG['save_path'])
                print(f"  → best 모델 저장 (val: {avg_val:.4f}, fpr: {fpr:.4f})")
            else:
                patience_counter += 1
                print(f"  → patience {patience_counter}/{CONFIG['patience']}")
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping! (epoch {epoch})")
                    break

    print(f"\nStage3 완료. best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_stage3()
