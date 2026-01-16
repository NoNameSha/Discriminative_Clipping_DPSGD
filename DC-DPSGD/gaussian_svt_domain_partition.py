"""
Gaussian SVT with Domain Partition (Algorithm 3)
Implementation of SVT-based Private Selection for identifying heavy-tailed samples
"""

import numpy as np
import torch
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed


class GaussianSVTDomainPartition:
    """
    Gaussian-based Sparse Vector Technique with domain partition for
    privately identifying heavy-tailed gradient samples.
    """

    def __init__(
        self,
        subspace_dim: int = 200,
        tail_proportion: float = 0.1,
        sigma1: float = None,
        sigma2: float = None,
        epsilon_tr: float = 0.4,
        delta_tr: float = 1e-5,
        theta: float = 2.0,
        device: str = 'cuda'
    ):
        """
        Initialize Gaussian SVT with domain partition.

        Args:
            subspace_dim: Dimension k of the projection subspace
            tail_proportion: Proportion p of heavy-tailed samples
            sigma1: Noise multiplier for threshold (computed from epsilon if None)
            sigma2: Noise multiplier for queries (computed from epsilon if None)
            epsilon_tr: Privacy budget for trace selection
            delta_tr: Privacy parameter delta
            theta: Sub-Weibull tail index for subspace construction
            device: Device to run computations on
        """
        self.k = subspace_dim
        self.p = tail_proportion
        self.epsilon_tr = epsilon_tr
        self.delta_tr = delta_tr
        self.theta = theta
        self.device = device

        # Compute noise multipliers from privacy budget
        # Following the paper: sigma1 = sqrt(2*log(1.25/delta1))/epsilon1
        epsilon1 = epsilon_tr / 2
        epsilon2 = epsilon_tr / 2
        delta1 = delta_tr / 2
        delta2 = delta_tr / 2

        self.sigma1 = sigma1 if sigma1 is not None else np.sqrt(2 * np.log(1.25 / delta1)) / epsilon1
        self.sigma2 = sigma2 if sigma2 is not None else np.sqrt(2 * np.log(1.25 / delta2)) / epsilon2

        # Subspace will be initialized when first called
        self.V_t_k = None

    def construct_subspace(self, grad_dim: int):
        """
        Construct projection subspace from sub-Weibull distribution.

        Args:
            grad_dim: Dimension d of gradients
        """
        # Sample k orthogonal vectors from sub-Weibull distribution
        # For simplicity, we use Gaussian sampling and orthogonalize
        V = torch.randn(grad_dim, self.k, device=self.device)

        # Orthogonalize using QR decomposition
        V, _ = torch.linalg.qr(V)

        self.V_t_k = V
        return V

    def compute_trace_score(self, normalized_grad: torch.Tensor) -> float:
        """
        Compute trace score: tr(V^T * g * g^T * V)

        Args:
            normalized_grad: Normalized gradient ĝ

        Returns:
            Trace score λ^tr
        """
        # λ^tr = tr(V^T * ĝ * ĝ^T * V) = ||V^T * ĝ||^2
        projected = self.V_t_k.T @ normalized_grad  # k-dim vector
        trace_score = torch.sum(projected ** 2).item()
        return trace_score

    def svt_on_partition(
        self,
        partition_samples: List[Tuple[int, torch.Tensor]],
        threshold_noisy: float,
        trace_sensitivity: float
    ) -> List[int]:
        """
        Run SVT on a single partition (executed in parallel for each partition).

        Args:
            partition_samples: List of (index, gradient) tuples for this partition
            threshold_noisy: Noisy threshold T̃
            trace_sensitivity: Sensitivity Δ of trace queries

        Returns:
            List of indices identified as heavy-tail samples
        """
        heavy_tail_indices = []

        for idx, gradient in partition_samples:
            # Normalize gradient
            grad_norm = torch.norm(gradient, p=2)
            if grad_norm > 0:
                normalized_grad = gradient / grad_norm
            else:
                normalized_grad = gradient

            # Compute trace score
            lambda_tr = self.compute_trace_score(normalized_grad)

            # Add query noise: ν_i ~ N(0, 4Δ²σ²_2)
            query_noise = np.random.normal(0, 2 * trace_sensitivity * self.sigma2)
            lambda_tr_noisy = lambda_tr + query_noise

            # Check if exceeds threshold
            if lambda_tr_noisy >= threshold_noisy:
                heavy_tail_indices.append(idx)
                break  # SVT stops after first above-threshold query

        return heavy_tail_indices

    def select_heavy_tail_samples(
        self,
        gradients: List[torch.Tensor],
        threshold: float = None,
        parallel: bool = True
    ) -> Tuple[List[int], List[int]]:
        """
        Main SVT-based private selection algorithm.

        Args:
            gradients: List of per-sample gradients
            threshold: Selection threshold T (auto-computed if None)
            parallel: Whether to run partitions in parallel

        Returns:
            (tail_indices, body_indices): Indices of heavy-tail and light-body samples
        """
        n = len(gradients)
        n_star = int(np.ceil(self.p * n))  # Number of partitions

        # Initialize subspace if not done
        if self.V_t_k is None:
            grad_dim = gradients[0].numel()
            self.construct_subspace(grad_dim)

        # Compute trace sensitivity: Δ = 1/k (due to normalization and projection)
        trace_sensitivity = 1.0 / self.k

        # Compute or use provided threshold
        if threshold is None:
            # Auto-compute threshold based on paper's formula
            threshold = np.sqrt(4 * np.log(1.25 / self.delta_tr) * np.log(n_star / self.delta_tr)) / (n_star * self.k * self.epsilon_tr)

        # Add threshold noise: ρ ~ N(0, Δ²σ²_1)
        threshold_noise = np.random.normal(0, trace_sensitivity * self.sigma1)
        threshold_noisy = threshold + threshold_noise

        # Partition dataset into n* disjoint subsets
        indices = np.arange(n)
        np.random.shuffle(indices)
        partitions = np.array_split(indices, n_star)

        # Prepare partition data
        partition_data = []
        for partition_indices in partitions:
            partition_samples = [(idx, gradients[idx]) for idx in partition_indices]
            partition_data.append(partition_samples)

        # Run SVT on each partition (in parallel if enabled)
        heavy_tail_indices = []

        if parallel and n_star > 1:
            # Parallel execution across partitions
            with ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self.svt_on_partition,
                        partition_samples,
                        threshold_noisy,
                        trace_sensitivity
                    )
                    for partition_samples in partition_data
                ]

                for future in as_completed(futures):
                    result = future.result()
                    heavy_tail_indices.extend(result)
        else:
            # Sequential execution
            for partition_samples in partition_data:
                result = self.svt_on_partition(
                    partition_samples,
                    threshold_noisy,
                    trace_sensitivity
                )
                heavy_tail_indices.extend(result)

        # Assign remaining samples to light-body region
        all_indices = set(range(n))
        tail_indices = set(heavy_tail_indices)
        body_indices = list(all_indices - tail_indices)

        return list(tail_indices), body_indices

    def __call__(
        self,
        gradients: List[torch.Tensor],
        threshold: float = None
    ) -> Tuple[List[int], List[int]]:
        """
        Convenience method to call select_heavy_tail_samples.
        """
        return self.select_heavy_tail_samples(gradients, threshold)
