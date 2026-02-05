from isaaclab_rl.rsl_rl.exporter import _TorchPolicyExporter, _OnnxPolicyExporter
import os, torch

def export_teacher_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    policy_exporter = _CRLTeacherTorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)

def export_teacher_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _CRLTeacherOnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)



class _CRLTeacherTorchPolicyExporter(_TorchPolicyExporter):
    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x, hist_encoding=True)
    
    def forward(self, x):
        return self.actor(self.normalizer(x), hist_encoding=True)
    
class _CRLTeacherOnnxPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x, hist_encoding=True ), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x), hist_encoding=True )

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor.in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )
