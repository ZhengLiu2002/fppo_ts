import torch.nn as nn
import torch
from rsl_rl.utils import resolve_nn_activation

class DefaultEstimator(nn.Module):
    """学习从本体观测恢复特权显式状态（如外界信息）的前馈 MLP。"""
    def __init__(self,  
                 num_prop,
                 num_priv_explicit,
                 num_priv_hurdles: int = 0,
                 hidden_dims=[256, 128, 64],
                 activation="elu",
                **kwargs):
        super(DefaultEstimator, self).__init__()

        self.input_dim = num_prop
        # Only estimate the "other" explicit states; hurdle semantics are not inferable from proprioception.
        self.output_dim = max(int(num_priv_explicit) - int(num_priv_hurdles or 0), 0)
        activation = resolve_nn_activation(activation)
        if self.output_dim > 0:
            estimator_layers = []
            estimator_layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
            estimator_layers.append(activation)
            for l in range(len(hidden_dims)):
                if l == len(hidden_dims) - 1:
                    estimator_layers.append(nn.Linear(hidden_dims[l], self.output_dim))
                else:
                    estimator_layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                    estimator_layers.append(activation)
            self.estimator = nn.Sequential(*estimator_layers)
        else:
            self.estimator = None
    
    def forward(self, input):
        if self.estimator is None:
            return input.new_zeros((input.shape[0], 0))
        return self.estimator(input)
    
    def inference(self, input):
        with torch.no_grad():
            if self.estimator is None:
                return input.new_zeros((input.shape[0], 0))
            return self.estimator(input)
