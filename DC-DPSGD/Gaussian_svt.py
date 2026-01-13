import torch
import math

def gaussian_svt(
    queries,                 # Tensor [n]
    epsilon,                 # privacy budget
    delta,                   # delta
    threshold,               # deterministic threshold T
    sigma_q,                 # std for query noise
    sigma_t,                 # std for threshold noise
):
    """
    Gaussian Sparse Vector Technique (SVT)

    Returns:
        idx_top: indices where noisy query exceeds noisy threshold
        idx_rest: remaining indices
    """

    device = queries.device
    n = queries.shape[0]

    # ---- noisy threshold ----
    T_tilde = threshold + torch.normal(
        mean=0.0,
        std=sigma_t,
        size=(1,),
        device=device
    )

    idx_top = []
    idx_rest = []

    # ---- sequential test ----
    for i in range(n):
        Qi_tilde = queries[i] + torch.normal(
            mean=0.0,
            std=sigma_q,
            size=(1,),
            device=device
        )

        if Qi_tilde >= T_tilde:
            idx_top.append(i)
        else:
            idx_rest.append(i)

    return idx_top, idx_rest, T_tilde.item()
