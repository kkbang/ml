"""
export_onnx.py — RoBERTa encoder만 ONNX export (inputs_embeds 입력)
DFG 노드 교체는 Python에서 처리, Transformer 12레이어만 TRT로 가속

실행: python export_onnx.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
import torch.nn.functional as F
from transformers import AutoModel

sys.path.append('/home/ngseokim/code-killr/parser')
from model import GraphCodeBERTEncoder
from dataset import TOTAL_LENGTH

MODEL_NAME  = 'microsoft/graphcodebert-base'
MODEL_PATH  = 'GCB_dfg_stage2.pt'
ONNX_PATH   = 'graphcodebert_encoder.onnx'
VERIFY_PATH = 'onnx_verify_result.txt'
OPSET       = 17
L           = TOTAL_LENGTH   # 320
D           = 768


# ── RoBERTa encoder 래퍼 ──────────────────────────────────────────────
class RoBERTaEncoderWrapper(nn.Module):
    """inputs_embeds (DFG 교체 완료) → [CLS] 임베딩"""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self,
                inputs_embeds:  torch.Tensor,   # (B, L, 768)
                attention_mask: torch.Tensor,   # (B, L) int64
                position_ids:   torch.Tensor,   # (B, L) int64
                ) -> torch.Tensor:              # (B, 768)
        token_type_ids = torch.zeros(
            inputs_embeds.shape[:2], dtype=torch.long,
            device=inputs_embeds.device,
        )
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        return outputs.last_hidden_state[:, 0, :]


# ── DFG 교체 함수 (Python side) ───────────────────────────────────────
def dfg_replace(input_ids, position_idx, attn_mask,
                word_embeddings, pad_token_id):
    nodes_mask    = position_idx.eq(0)
    token_mask    = position_idx.ge(2)
    inputs_embeds = word_embeddings(input_ids)

    nodes_to_token_mask = (
        nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
    ).float()
    nodes_to_token_mask = nodes_to_token_mask / (
        nodes_to_token_mask.sum(-1, keepdim=True) + 1e-10
    )
    avg_embeddings = torch.einsum(
        "abc,acd->abd", nodes_to_token_mask, inputs_embeds
    )
    inputs_embeds = (
        inputs_embeds * (~nodes_mask)[:, :, None]
        + avg_embeddings * nodes_mask[:, :, None]
    )
    attention_mask_1d = input_ids.ne(pad_token_id).long()
    return inputs_embeds, attention_mask_1d


def load_components():
    base_encoder    = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model           = GraphCodeBERTEncoder(base_encoder)
    ckpt            = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt))
    model.eval()
    wrapper         = RoBERTaEncoderWrapper(model.encoder)
    word_embeddings = model.encoder.embeddings.word_embeddings
    pad_token_id    = model.encoder.config.pad_token_id
    return wrapper, word_embeddings, pad_token_id


def export_onnx(wrapper):
    print(f"ONNX export 시작 (opset={OPSET}, L={L}, D={D})")
    B = 2
    dummy_embeds = torch.randn(B, L, D)
    dummy_attn   = torch.ones(B, L, dtype=torch.long)
    dummy_pos    = torch.ones(B, L, dtype=torch.long)
    dummy_pos[:, :L//2] = torch.arange(2, L//2 + 2)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_embeds, dummy_attn, dummy_pos),
            ONNX_PATH,
            input_names=['inputs_embeds', 'attention_mask', 'position_ids'],
            output_names=['cls_output'],
            dynamic_axes={
                'inputs_embeds':  {0: 'batch'},
                'attention_mask': {0: 'batch'},
                'position_ids':   {0: 'batch'},
                'cls_output':     {0: 'batch'},
            },
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"  → {ONNX_PATH} 저장 완료")


def verify_onnx(wrapper, word_embeddings, pad_token_id):
    print("\nONNX 검증 중...")
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print("  ✓ ONNX 구조 검증 통과")

    B = 1
    input_ids             = torch.randint(3, 50000, (B, L), dtype=torch.long)
    input_ids[:, 0]       = 0
    input_ids[:, L//2]    = 2
    input_ids[:, L//2+1:] = 1
    position_ids          = torch.ones(B, L, dtype=torch.long)
    position_ids[:, :L//2] = torch.arange(2, L//2 + 2)
    attn_mask_3d          = torch.zeros(B, L, L, dtype=torch.bool)
    attn_mask_3d[:, :L//2, :L//2] = True

    with torch.no_grad():
        inputs_embeds, attn_1d = dfg_replace(
            input_ids, position_ids, attn_mask_3d,
            word_embeddings, pad_token_id,
        )
        pt_emb = F.normalize(wrapper(inputs_embeds, attn_1d, position_ids), dim=1).numpy()

    providers = (
        ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers()
        else ['CPUExecutionProvider']
    )
    print(f"  ORT provider: {providers[0]}")
    sess    = ort.InferenceSession(ONNX_PATH, providers=providers)
    ort_out = sess.run(['cls_output'], {
        'inputs_embeds':  inputs_embeds.numpy(),
        'attention_mask': attn_1d.numpy(),
        'position_ids':   position_ids.numpy(),
    })[0]

    if np.isnan(ort_out).any():
        print("  ⚠️  ORT NaN → TRT GPU에서 재확인 필요")
        open(VERIFY_PATH, 'w').write("passed: False (NaN)\n")
        return True

    ort_emb  = ort_out / (np.linalg.norm(ort_out, axis=1, keepdims=True) + 1e-8)
    max_diff = np.abs(pt_emb - ort_emb).max()
    passed   = max_diff < 1e-3
    print(f"  PyTorch vs ORT 최대 오차: {max_diff:.6f}")
    print(f"  {'✓ 검증 통과' if passed else '✗ 오차 초과'}")
    open(VERIFY_PATH, 'w').write(
        f"ONNX_PATH: {ONNX_PATH}\nmax_diff: {max_diff:.6f}\npassed: {passed}\n"
    )
    return passed


if __name__ == '__main__':
    print("=== GraphCodeBERT Split ONNX Export ===\n")
    wrapper, word_embeddings, pad_token_id = load_components()
    print(f"모델 로드 완료: {MODEL_PATH}")
    export_onnx(wrapper)
    ok = verify_onnx(wrapper, word_embeddings, pad_token_id)
    print(f"\n{'완료. 다음 단계: python convert_trt.py' if ok else 'Export 실패.'}")
