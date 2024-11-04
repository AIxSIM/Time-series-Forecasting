import torch.nn.functional as F
import torch

def contrastive_loss(z_i, z_j, temperature=0.5):
    """Compute contrastive loss for a batch of representations."""
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Similarity scores
    logits = torch.matmul(z_i, z_j.T) / temperature
    labels = torch.arange(logits.shape[0]).long().to(logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss