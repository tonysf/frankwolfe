"""Withheld correspondence diagnostics; no random draws or optimizer mutation."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def load_correspondence(filename, expected_sha256, n, m, device):
    if filename is None:
        if expected_sha256 is not None:
            raise ValueError('A correspondence hash requires a file.')
        return None
    path = Path(filename)
    if not expected_sha256 or hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError('Correspondence SHA256 mismatch.')
    with np.load(path, allow_pickle=False) as data:
        forward, inverse = data['forward'], data['inverse']
        if n != m or forward.shape != (n,) or inverse.shape != (m,):
            raise ValueError('Expected a bijective correspondence for a square paired instance.')
        if forward.dtype.kind not in 'iu' or inverse.dtype.kind not in 'iu':
            raise ValueError('Correspondence indices must be integers.')
        if not np.array_equal(np.sort(forward), np.arange(n)) or not np.array_equal(inverse[forward], np.arange(n)):
            raise ValueError('Correspondence is not a mutually inverse permutation.')
        return (torch.as_tensor(forward.astype(np.int64), device=device),
                torch.as_tensor(inverse.astype(np.int64), device=device))


@torch.no_grad()
def correspondence_metrics(plan, correspondence, chunk=256):
    """Symmetric true-match ranks/top1, with uniform credit for exact ties.

    Rank fraction is (strictly greater entries + half of tied wrong entries)/(n-1).
    This is a coupling-row rank metric, not embedding-space FOSCTTM.
    Initial uniform coupling has rank fraction .5 and top1 accuracy 1/n.
    True-pair mass is the sum of the coupling on the known permutation.
    """
    if chunk < 1:
        raise ValueError('chunk must be positive.')
    forward, inverse = correspondence
    if plan.ndim != 2 or plan.shape[0] != plan.shape[1] or len(forward) != len(plan):
        raise ValueError('Plan/correspondence shape mismatch.')
    n = plan.shape[0]
    accuracies, ranks = [], []
    for matrix, mapping in ((plan, forward), (plan.T, inverse)):
        accuracy = torch.zeros((), device=plan.device, dtype=torch.float64)
        rank = torch.zeros_like(accuracy)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            rows = matrix[start:stop]
            truth = rows[torch.arange(stop-start, device=plan.device), mapping[start:stop]]
            equal = (rows == truth[:, None]).sum(1)
            higher = (rows > truth[:, None]).sum(1)
            accuracy += ((higher == 0).double() / equal).sum()
            if n > 1:
                rank += ((higher.double() + .5 * (equal.double()-1)) / (n-1)).sum()
        accuracies.append((accuracy/n).item()); ranks.append((rank/n).item())
    mass = plan[torch.arange(n, device=plan.device), forward].double().sum().item()
    return dict(correspondence_top1_accuracy=sum(accuracies)/2,
                correspondence_rank_fraction=sum(ranks)/2,
                correspondence_true_pair_mass=mass,
                correspondence_forward_top1_accuracy=accuracies[0],
                correspondence_inverse_top1_accuracy=accuracies[1])
