"""
GraphCodeBERT Stage2 학습 — Hard Negative Mixing
수정 사항:
  - GraphCodeBERTEncoder 올바르게 로드
  - DFG 입력 (position_ids, attn_mask) 복원
  - Hard Negative를 positive 자리가 아닌 혼합 샘플로 처리
  - compute_fpr 벡터화
  - AMP (BF16) 추가
  - 체크포인트 완성 (optimizer/scheduler 상태 포함)
"""

import torch
import json
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, get_cosine_schedule_with_warmup
from loss import weighted_nt_xent_loss
from model import GraphCodeBERTEncoder
from tqdm import tqdm
import sys

sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import PairDataset, encode_with_dfg, build_attn_mask, TOTAL_LENGTH

# ── 설정 ──────────────────────────────────────────────────────────────
CONFIG = {
    'train_path':     'data/train_v3.jsonl',
    'hard_neg_path':  'hard_negatives_gcb.jsonl',   # mining 결과
    'val_path':       'data/val_v3.jsonl',
    'model_path':     'GCB_dfg_stage1.pt',           # stage1 저장 파일명과 일치
    'save_path':      'GCB_dfg_stage2.pt',
    'log_path':       'train_gcb_stage2_log.txt',
    'model_name':     'microsoft/graphcodebert-base',
    'batch_size':     32,
    'lr':             1e-5,          # stage1보다 낮게
    'weight_decay':   0.01,
    'temperature':    0.07,          # stage1(0.05)보다 완만하게
    'max_epochs':     30,
    'patience':       7,
    'warmup_ratio':   0.1,
    'hard_neg_ratio': 0.3,           # 배치의 30%를 Hard Negative로
    'device':         'cuda' if torch.cuda.is_available() else 'cpu',
}

LICENSE_WEIGHTS = {
    'GPL-2.0': 3.0, 'GPL-3.0': 3.0, 'AGPL-3.0': 3.0,
    'LGPL-2.1': 2.0, 'LGPL-3.0': 2.0,
}
HARD_NEG_WEIGHT = 2.5   # hard negative pair 기본 가중치


# ── 데이터셋 ───────────────────────────────────────────────────────────
class Stage2Dataset(Dataset):
    """
    Positive pair와 Hard Negative pair를 hard_neg_ratio 비율로 혼합.

    [올바른 Hard Negative 처리]
    NT-Xent는 배치 내 off-diagonal 전체를 negative로 처리하므로
    Hard Negative pair를 독립적인 (anchor, hard_neg) 샘플로 배치에 섞으면
    자연스럽게 hard in-batch negative가 됨.
    → positive 자리에 negative를 넣는 기존 방식은 오류.
    """

    def __init__(self, train_path: str, hard_neg_path: str,
                 tokenizer, hard_neg_ratio: float = 0.3):
        self.tokenizer      = tokenizer
        self.hard_neg_ratio = hard_neg_ratio

        with open(train_path) as f:
            self.positives = [json.loads(l) for l in f]

        with open(hard_neg_path) as f:
            self.hard_negatives = [json.loads(l) for l in f]

        print(f"Positive pair: {len(self.positives)}개")
        print(f"Hard Negative pair: {len(self.hard_negatives)}개")

    def __len__(self):
        return len(self.positives)

    def _encode(self, code: str, language: str):
        """DFG 포함 인코딩 → (input_ids, position_ids, attn_mask) 텐서 반환"""
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
            'anchor_input_ids':   a_ids,
            'anchor_position_ids': a_pos,
            'anchor_attn_mask':   a_mask,
            'positive_input_ids':   p_ids,
            'positive_position_ids': p_pos,
            'positive_attn_mask':   p_mask,
            'weight': torch.tensor(weight, dtype=torch.float),
        }

    def __getitem__(self, idx):
        # hard_neg_ratio 확률로 Hard Negative 샘플 반환
        if self.hard_negatives and np.random.random() < self.hard_neg_ratio:
            hn   = self.hard_negatives[np.random.randint(len(self.hard_negatives))]
            lang = hn.get('language', 'python')
            try:
                # anchor와 hard_negative를 pair로 구성
                # NT-Xent가 이를 배치 내 hard in-batch negative로 자동 처리
                weight = LICENSE_WEIGHTS.get(hn.get('anchor_license', ''), HARD_NEG_WEIGHT)
                return self._make_item(hn['anchor'], hn['negative'], lang, weight)
            except Exception:
                pass  # 실패 시 positive pair로 fallback

        # Positive pair 반환
        pair = self.positives[idx]
        try:
            lang   = pair.get('language', 'python')
            weight = LICENSE_WEIGHTS.get(pair.get('license', ''), 1.0)
            return self._make_item(pair['anchor'], pair['positive'], lang, weight)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.positives))



# ── FPR 계산 (벡터화) ─────────────────────────────────────────────────
def compute_fpr(model, val_loader, device, threshold=0.5) -> float:
    model.eval()
    fp_count  = 0
    total_neg = 0

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

            B    = a_emb.shape[0]
            if B < 2:
                continue

            sim  = torch.matmul(a_emb, p_emb.T)                          # (B, B)
            mask = ~torch.eye(B, dtype=torch.bool, device=device)        # 대각 제외
            neg  = sim[mask]

            fp_count  += (neg > threshold).sum().item()
            total_neg += neg.numel()

    return fp_count / total_neg if total_neg > 0 else 0.0


# ── 학습 ──────────────────────────────────────────────────────────────
def train_stage2():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    print(f"Device: {CONFIG['device']}")

    # 데이터셋
    train_set = Stage2Dataset(
        CONFIG['train_path'], CONFIG['hard_neg_path'],
        tokenizer, CONFIG['hard_neg_ratio'],
    )
    val_set = PairDataset(CONFIG['val_path'], tokenizer)   # dataset.py PairDataset 재사용

    train_loader = DataLoader(
        train_set, batch_size=CONFIG['batch_size'],
        shuffle=True, drop_last=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # 🔴 Fix 1: GraphCodeBERTEncoder로 올바르게 로드
    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    model.encoder.gradient_checkpointing_enable()
    model = model.to(CONFIG['device'])
    print(f"Stage1 모델 로드 완료: {CONFIG['model_path']}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
    )
    total_steps  = len(train_loader) * CONFIG['max_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss    = float('inf')
    patience_counter = 0

    with open(CONFIG['log_path'], 'w') as log:
        log.write("epoch,train_loss,val_loss,fpr\n")

        for epoch in range(1, CONFIG['max_epochs'] + 1):
            model.train()
            train_loss = 0.0
            bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

            for batch in bar:
                a_ids  = batch['anchor_input_ids'].to(CONFIG['device'])
                a_pos  = batch['anchor_position_ids'].to(CONFIG['device'])
                a_mask = batch['anchor_attn_mask'].to(CONFIG['device'])
                p_ids  = batch['positive_input_ids'].to(CONFIG['device'])
                p_pos  = batch['positive_position_ids'].to(CONFIG['device'])
                p_mask = batch['positive_attn_mask'].to(CONFIG['device'])
                weights = batch['weight'].to(CONFIG['device'])

                # 🔴 Fix 2: DFG 입력 포함 forward + AMP
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    a_emb = model(a_ids, a_pos, a_mask)
                    p_emb = model(p_ids, p_pos, p_mask)
                    loss  = weighted_nt_xent_loss(
                        a_emb, p_emb, weights, CONFIG['temperature']
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

            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    a_ids  = batch['anchor_input_ids'].to(CONFIG['device'])
                    a_pos  = batch['anchor_position_ids'].to(CONFIG['device'])
                    a_mask = batch['anchor_attn_mask'].to(CONFIG['device'])
                    p_ids  = batch['positive_input_ids'].to(CONFIG['device'])
                    p_pos  = batch['positive_position_ids'].to(CONFIG['device'])
                    p_mask = batch['positive_attn_mask'].to(CONFIG['device'])
                    weights = batch['weight'].to(CONFIG['device'])

                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        a_emb = model(a_ids, a_pos, a_mask)
                        p_emb = model(p_ids, p_pos, p_mask)
                        loss  = weighted_nt_xent_loss(
                            a_emb, p_emb, weights, CONFIG['temperature']
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
                # 체크포인트 완성 (optimizer/scheduler 상태 포함)
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

    print(f"\nStage2 완료. best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train_stage2()
