import torch
import torch.nn as nn


class GraphCodeBERTEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, position_idx, attn_mask):
        """
        input_ids:    (B, L)
        position_idx: (B, L) — 코드 토큰: >=2, DFG 노드: 0, 패딩: 1
        attn_mask:    (B, L, L) — DFG avg_embeddings 계산에만 사용
        """
        # ── DFG 노드 임베딩 교체 (공식 model.py 동일) ──────
        nodes_mask = position_idx.eq(0)   # DFG 노드
        token_mask = position_idx.ge(2)   # 코드 토큰

        inputs_embeddings = self.encoder.embeddings.word_embeddings(input_ids)

        # DFG 노드 임베딩 → 연결된 코드 토큰 평균으로 교체
        nodes_to_token_mask = (
            nodes_mask[:, :, None]
            & token_mask[:, None, :]
            & attn_mask
        ).float()
        nodes_to_token_mask = nodes_to_token_mask / (
            nodes_to_token_mask.sum(-1, keepdim=True) + 1e-10
        )
        avg_embeddings    = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = (
            inputs_embeddings * (~nodes_mask)[:, :, None]
            + avg_embeddings  *   nodes_mask [:, :, None]
        )

        # 1D 패딩 마스크
        attention_mask_1d = input_ids.ne(self.encoder.config.pad_token_id).long()

        # RoBERTa forward (1D mask 사용)
        outputs = self.encoder(
            inputs_embeds=inputs_embeddings,
            attention_mask=attention_mask_1d,
            position_ids=position_idx,
            token_type_ids=position_idx.eq(-1).long(),
        )

        return outputs.last_hidden_state[:, 0, :]

    def get_embedding(self, input_ids, position_idx, attn_mask):
        with torch.no_grad():
            return self.forward(input_ids, position_idx, attn_mask)
