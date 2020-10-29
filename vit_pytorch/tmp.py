# MLP
if classifier == 'token':
    self.mlp = nn.Sequential(
        nn.Linear(dim, mlp_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, mlp_dim),
        nn.Dropout(dropout_rate)
    )


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self, dim, mlp_dim, out_dim, dropout_rate):
        super().__init__()
        self
    
    def apply(self,
            inputs,
            mlp_dim,
            dtype=jnp.float32,
            out_dim=None,
            dropout_rate=0.1,
            deterministic=True,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.normal(stddev=1e-6)):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
        return output
