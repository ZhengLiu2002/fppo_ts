
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from .feature_extractors import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from .actor_critic_with_encoder import ActorCriticRMA
from copy import deepcopy  

class DistillationWithExtractor():
    """视觉蒸馏算法包装：用深度编码器+学生 actor 拟合教师动作。"""
    policy: ActorCriticRMA
    def __init__(
        self,
        policy,
        estimator, 
        estimator_paras,
        learning_rate,
        depth_encoder_cfg,
        policy_cfg,
        device,
        max_grad_norm=1.0,
        multi_gpu_cfg: dict | None = None,
        ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1
        # teacher policy + 结构化状态估计器
        self.estimator: nn.Module = estimator
        self.priv_states_dim = estimator_paras["num_priv_explicit"]
        self.num_prop = estimator_paras["num_prop"]
        self.num_scan = estimator_paras["num_scan"]
        self.policy = policy
        estimator_params = list(self.estimator.parameters())
        self.estimator_optimizer = (
            optim.Adam(estimator_params, lr=estimator_paras["learning_rate"]) if estimator_params else None
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.rnd = None  # TODO: remove when runner has a proper base class

        depth_backbone_class = eval(depth_encoder_cfg.pop("backbone_class_name"))
        depth_backbone : DepthOnlyFCBackbone58x87 = depth_backbone_class( 
            policy_cfg["scan_encoder_dims"][-1], depth_encoder_cfg["hidden_dims"],
        )
        depth_encoder_class = eval(depth_encoder_cfg.pop("encoder_class_name"))
        depth_encoder: RecurrentDepthBackbone = depth_encoder_class(depth_backbone, depth_encoder_cfg).to(device)
        depth_actor: ActorCriticRMA = deepcopy(policy.actor)
        self.learning_rate = depth_encoder_cfg["learning_rate"]
        self.max_grad_norm = max_grad_norm
        self.depth_encoder = depth_encoder
        self.depth_encoder_cfg = depth_encoder_cfg
        self.depth_actor = depth_actor
        self.depth_actor_optimizer = optim.Adam([*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=depth_encoder_cfg["learning_rate"])

    def broadcast_parameters(self):
        """同步教师/估计器/深度编码器/学生 actor 参数到所有 GPU。"""
        if not self.is_multi_gpu:
            return
        state_to_sync = {
            "policy": self.policy.state_dict(),
            "estimator": self.estimator.state_dict(),
            "depth_encoder": self.depth_encoder.state_dict(),
            "depth_actor": self.depth_actor.state_dict(),
        }
        obj_list = [state_to_sync]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        synced_state = obj_list[0]
        self.policy.load_state_dict(synced_state["policy"])
        self.estimator.load_state_dict(synced_state["estimator"])
        self.depth_encoder.load_state_dict(synced_state["depth_encoder"])
        self.depth_actor.load_state_dict(synced_state["depth_actor"])

    def reduce_parameters(self):
        """跨 GPU 求平均梯度，确保分布式更新一致。"""
        if not self.is_multi_gpu:
            return
        modules = [self.policy, self.estimator, self.depth_encoder, self.depth_actor]
        for module in modules:
            for param in module.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
                    param.grad /= self.gpu_world_size

    def update_depth_actor(self, actions_buffer, yaws_buffer):
        """学生 actor 损失：逼近教师动作 + 航向校正。"""
        depth_actor_loss = (actions_buffer).norm(p=2, dim=1).mean()
        yaw_loss = (yaws_buffer).norm(p=2, dim=1).mean()

        loss = depth_actor_loss + yaw_loss
        self.depth_actor_optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.depth_encoder.parameters(), self.max_grad_norm)
        self.depth_actor_optimizer.step()
        loss_dict = {
            "depth_actor_loss": depth_actor_loss.item(),
            "yaw_loss": yaw_loss.item(),
            "total_loss": loss.item(),
        }
        return loss_dict

    def broadcast_parameters(self):
        """仅同步 policy 参数的简化版本（覆盖上方函数，保留兼容性）。"""
        # obtain the model parameters on current GPU
        model_params = [self.policy.state_dict()]
        # broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # load the model parameters on all GPUs from source GPU
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        """仅对 policy 梯度做平均（覆盖上方函数，历史遗留接口）。"""
        # Create a tensor to store the gradients
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                # copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # update the offset for the next parameter
                offset += numel
