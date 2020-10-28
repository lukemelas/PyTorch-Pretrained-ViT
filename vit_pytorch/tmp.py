# MLP
if classifier == 'token':
    self.mlp = nn.Sequential(
        nn.Linear(dim, mlp_dim),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, mlp_dim),
        nn.Dropout(dropout_rate)
    )
