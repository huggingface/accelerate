from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, ACT2FN
import torch.nn as nn
from axonn.intra_layer import Linear
import torch

def modified_attention_init(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
    super(OPTAttention, self).__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.dropout = dropout
    self.head_dim = embed_dim // num_heads

    if (self.head_dim * num_heads) != self.embed_dim:
        raise ValueError(
            f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
            f" and `num_heads`: {num_heads})."
        )
    self.scaling = self.head_dim**-0.5
    self.is_decoder = is_decoder

    self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
    self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
    self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
    self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

def modified_decoder_init(self, config):
    super(OPTDecoderLayer, self).__init__()
    self.embed_dim = config.hidden_size
    self.self_attn = OPTAttention(
        embed_dim=self.embed_dim,
        num_heads=config.num_attention_heads,
        dropout=config.attention_dropout,
        is_decoder=True,
        bias=config.enable_bias,
    )
    self.do_layer_norm_before = config.do_layer_norm_before
    self.dropout = config.dropout
    self.activation_fn = ACT2FN[config.activation_function]

    self.self_attn_layer_norm = nn.LayerNorm(
        self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
    )
    self.fc1 = Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias)
    self.fc2 = Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias)
    self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)


def monkey_patch_opt_with_axonn():
    OPTAttention.__init__ = modified_attention_init
    OPTDecoderLayer.__init__ = modified_decoder_init
