import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from model import GraphCodeBERTEncoder
from dataset import PairDataset
from loss import weighted_nt_xent_loss
from tqdm import tqdm
import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')

sys.path.append('/home/ngseokim/code-killr/parser')

CONFIG = {
    'train_path':   'data/train_v3.jsonl',
    'val_path':     'data/val_v3.jsonl',
    'save_path':    'GCB_dfg_stage1.pt',
    'log_path':     'train_gcb_dfg_log.txt',
    'model_name':   'microsoft/graphcodebert-base',
    'batch_size':   64,
    'lr':           2e-5,
    'weight_decay': 0.01,
    'temperature':  0.05,
    'max_epochs':   30,
    'patience':     5,
    'warmup_ratio': 0.1,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])


def compute_fpr(model, val_loader, device, threshold=0.5):
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
            B   = a_emb.shape[0]
            sim = torch.matmul(a_emb, p_emb.T)
            for i in range(B):
                for j in range(B):
                    if i != j:
                        total_neg += 1
                        if sim[i, j].item() > threshold:
                            fp_count += 1
    return fp_count / total_neg if total_neg > 0 else 0.0


def train():
    print(f"Device: {CONFIG['device']}")
    print(f"모델: {CONFIG['model_name']} (DFG avg_embeddings + position_ids)")

    train_set = PairDataset(CONFIG['train_path'], tokenizer)
    val_set   = PairDataset(CONFIG['val_path'],   tokenizer)

    train_loader = DataLoader(
        train_set, batch_size=CONFIG['batch_size'],
        shuffle=True, drop_last=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_set, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4,
    )

    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    model.encoder.gradient_checkpointing_enable()
    model   = model.to(CONFIG['device'])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'],
    )

    total_steps  = len(train_loader) * CONFIG['max_epochs']
    warmup_steps = int(total_steps * CONFIG['warmup_ratio'])
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss    = float('inf')
    patience_counter = 0

    with open(CONFIG['log_path'], 'w') as log:
        log.write("epoch,train_loss,val_loss,fpr\n")

        for epoch in range(1, CONFIG['max_epochs'] + 1):
            model.train()
            train_loss = 0.0
            bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)

            for batch in bar:
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
                weights = batch['weight'].to(CONFIG['device'])
                loss    = weighted_nt_xent_loss(a_emb, p_emb, weights, CONFIG['temperature'])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train = train_loss / len(train_loader)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
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
                    weights = batch['weight'].to(CONFIG['device'])
                    loss    = weighted_nt_xent_loss(a_emb, p_emb, weights, CONFIG['temperature'])
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            fpr     = compute_fpr(model, val_loader, CONFIG['device'])

            print(f"[Epoch {epoch}] train: {avg_train:.4f} | val: {avg_val:.4f} | fpr: {fpr:.4f}")
            log.write(f"{epoch},{avg_train:.4f},{avg_val:.4f},{fpr:.4f}\n")
            log.flush()

            if avg_val < best_val_loss:
                best_val_loss    = avg_val
                patience_counter = 0
                torch.save(model.state_dict(), CONFIG['save_path'])
                print(f"  → best 모델 저장 (val: {avg_val:.4f}, fpr: {fpr:.4f})")
            else:
                patience_counter += 1
                print(f"  → patience {patience_counter}/{CONFIG['patience']}")
                if patience_counter >= CONFIG['patience']:
                    print(f"Early stopping! (epoch {epoch})")
                    break

    print(f"\nStage1 완료. best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    train()
