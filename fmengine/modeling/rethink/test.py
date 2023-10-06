import torch
from fmengine.modeling.rethink.blocks import CausalSelfAttention
from fmengine.modeling.rethink.config import RethinkModelConfig

def test_causal_self_attention():
    # Define a configuration object
    config = RethinkModelConfig()

    # Create an instance of the CausalSelfAttention class
    causal_self_attention = CausalSelfAttention(config).to("cuda")

    # Define some input data
    x = torch.randn(32, 2048, 1024).to("cuda")

    # Test the forward method
    y = causal_self_attention(x)

    # Check the output shape
    assert y.shape == x.shape

if __name__=="__main__":
    test_causal_self_attention()