import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
import fmengine.mpu as mpu
from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear

class TensorParallelLlamaMLP(nn.Module):
    def __init__(
        self,
        args,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            process_group=None,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            process_group=None,
            bias=False,
        )
        self.up_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            process_group=None,
            bias=False,
        )
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)[0]) * self.up_proj(x)[0])[0]

class TensorParallelLlamaAttention(LlamaAttention):
    def __init__(self, args, config: LlamaConfig, no_reduce=False):
        super().__init__(config)
        self.q_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_heads * self.head_dim,
            process_group=None,
            bias=False,
        )
        self.k_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            process_group=None,
            bias=False,
        )
        self.v_proj = ColumnParallelLinear(
            in_features=self.hidden_size,
            out_features=self.num_key_value_heads * self.head_dim,
            process_group=None,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            in_features=self.num_heads * self.head_dim,
            out_features=self.hidden_size,
            process_group=None,
            bias=False,
        )