from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch


def conjugate_gradients(
    fisher_product,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Solve ``A x = b`` with conjugate gradients using an implicit matrix product."""

    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(max(int(num_steps), 1)):
        vector_z = fisher_product(vector_p)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(torch.clamp_min(new_rdotr, 0.0)) < residual_tol:
            break
        beta = new_rdotr / (rdotr + eps)
        vector_p = vector_r + beta * vector_p
        rdotr = new_rdotr

    return vector_x


def trainable_parameters(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    """Materialize trainable parameters while preserving iteration order."""

    return [parameter for parameter in parameters if parameter.requires_grad]


def flatten_tensor_sequence(
    tensors: Sequence[torch.Tensor | None],
    references: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    """Flatten tensors, using zero tensors when the corresponding entry is ``None``."""

    flat_tensors: list[torch.Tensor] = []
    for tensor, reference in zip(tensors, references, strict=True):
        if tensor is None:
            flat_tensors.append(torch.zeros_like(reference, memory_format=torch.contiguous_format).view(-1))
        else:
            flat_tensors.append(tensor.contiguous().view(-1))
    if not flat_tensors:
        raise AssertionError("No tensors were provided.")
    return torch.cat(flat_tensors)


def get_flat_params_from(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten trainable parameters into a single vector."""

    params = trainable_parameters(parameters)
    if not params:
        raise AssertionError("No trainable parameters were found.")
    return torch.cat([parameter.data.view(-1) for parameter in params])


def get_flat_gradients_from(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten gradients into a single vector, filling missing grads with zeros."""

    params = trainable_parameters(parameters)
    if not params:
        raise AssertionError("No trainable parameters were found.")
    grads = [parameter.grad for parameter in params]
    return flatten_tensor_sequence(grads, params)


def set_param_values_to_parameters(
    parameters: Iterable[torch.nn.Parameter],
    values: torch.Tensor,
) -> None:
    """Copy a flattened parameter vector back into an ordered parameter list."""

    params = trainable_parameters(parameters)
    if not params:
        raise AssertionError("No trainable parameters were found.")

    offset = 0
    for parameter in params:
        numel = parameter.numel()
        new_values = values[offset : offset + numel].view_as(parameter)
        parameter.data.copy_(new_values)
        offset += numel

    if offset != len(values):
        raise AssertionError(f"Lengths do not match: {offset} vs. {len(values)}")


class Lagrange:
    """Naive Lagrange multiplier helper ported from OmniSafe."""

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        lagrangian_upper_bound: float | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.cost_limit = float(cost_limit)
        self.lambda_lr = float(lambda_lr)
        self.lagrangian_upper_bound = (
            float(lagrangian_upper_bound) if lagrangian_upper_bound is not None else None
        )

        init_value = max(float(lagrangian_multiplier_init), 0.0)
        self.lagrangian_multiplier = torch.nn.Parameter(
            torch.as_tensor(init_value, dtype=torch.float32, device=device),
            requires_grad=True,
        )

        if not hasattr(torch.optim, lambda_optimizer):
            raise AttributeError(f"Optimizer={lambda_optimizer} not found in torch.optim.")
        optimizer_class = getattr(torch.optim, lambda_optimizer)
        self.lambda_optimizer = optimizer_class([self.lagrangian_multiplier], lr=self.lambda_lr)

    def compute_lambda_loss(self, mean_ep_cost: float) -> torch.Tensor:
        """Compute the dual loss for the Lagrange multiplier."""

        return -self.lagrangian_multiplier * (float(mean_ep_cost) - self.cost_limit)

    def update_lagrange_multiplier(self, mean_ep_cost: float) -> None:
        """Apply one optimizer step on the dual variable."""

        self.lambda_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(mean_ep_cost)
        lambda_loss.backward()
        self.lambda_optimizer.step()

        upper = self.lagrangian_upper_bound
        if upper is None:
            self.lagrangian_multiplier.data.clamp_(min=0.0)
        else:
            self.lagrangian_multiplier.data.clamp_(0.0, upper)

    def state_dict(self) -> dict:
        """Serialize the multiplier and its optimizer state."""

        return {
            "cost_limit": self.cost_limit,
            "lambda_lr": self.lambda_lr,
            "lagrangian_upper_bound": self.lagrangian_upper_bound,
            "lagrangian_multiplier": self.lagrangian_multiplier.detach().clone(),
            "lambda_optimizer": self.lambda_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the multiplier and its optimizer state."""

        if not state_dict:
            return
        multiplier = state_dict.get("lagrangian_multiplier", None)
        if multiplier is not None:
            self.lagrangian_multiplier.data.copy_(multiplier.to(self.lagrangian_multiplier.device))
        optimizer_state = state_dict.get("lambda_optimizer", None)
        if optimizer_state is not None:
            self.lambda_optimizer.load_state_dict(optimizer_state)
