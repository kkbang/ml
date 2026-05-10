import torch
import torch.nn.functional as F


def weighted_nt_xent_loss(
    anchors:     torch.Tensor,  # (B, 768)
    positives:   torch.Tensor,  # (B, 768)
    weights:     torch.Tensor,  # (B,)
    temperature: float = 0.05,
) -> torch.Tensor:

    B = anchors.shape[0]

    # L2 정규화
    anchors   = F.normalize(anchors,   dim=1)
    positives = F.normalize(positives, dim=1)

    # anchor + positive 합치기 (2B, 768)
    embeddings = torch.cat([anchors, positives], dim=0)

    # 유사도 행렬 (2B, 2B)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # 자기 자신 마스킹
    mask = torch.eye(2 * B, dtype=torch.bool, device=anchors.device)
    sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))

    # Positive 쌍 인덱스
    # anchor i → positive i+B
    # positive i+B → anchor i
    labels = torch.cat([
        torch.arange(B, 2 * B),
        torch.arange(0, B),
    ]).to(anchors.device)

    # Loss 계산
    loss_per_sample = F.cross_entropy(sim_matrix, labels, reduction='none')

    # 라이선스 가중치 적용
    weights_doubled = torch.cat([weights, weights]).to(anchors.device)
    weighted_loss = (loss_per_sample * weights_doubled).mean()

    return weighted_loss
