import torch
import math

def gaussian_svt(
    queries,        # Tensor [n]
    n_star,         # cutoff n*
    sigma_q,        # std of query noise
    sigma_t,        # std of threshold noise
):
    """
    Gaussian SVT aligned with Algorithm 3 (SVT-based Private Subspace Identification)

    Returns:
        idx_tail: selected heavy-tail indices (S^tail)
        idx_body: remaining indices (S^body)
    """

    device = queries.device
    n = queries.shape[0]

    idx_tail = []
    idx_body = []
    n_c = 0

    rho = torch.normal(
        mean=0.0,
        std=sigma_t,
        size=(1,),
        device=device
    )
    T_tilde = 1*0.9 + rho

    for i in range(n):

        v_i = torch.normal(
            mean=0.0,
            std=sigma_q,
            size=(1,),
            device=device
        )
        Qi_tilde = queries[i] + v_i

        if Qi_tilde >= T_tilde:
            idx_tail.append(i)
            n_c += 1

            if n_c >= n_star:
                break

            rho = torch.normal(
                mean=0.0,
                std=sigma_t,
                size=(1,),
                device=device
            )
            T_tilde = threshold + rho
        else:
            idx_body.append(i)

    # ---- remaining samples → body ----
    for j in range(i + 1, n):
        idx_body.append(j)

    return idx_tail, idx_body

