from torch import nn


class NEndModel(nn.Module):
    """
    Simple MLP Model (Not-End-to-End-Model) with a single layer.
    """

    def __init__(
        self,
        input_shape: int,
        out: int,
        dropout_rate: float
    ):
        super().__init__()
        self.model_name = "simple_forward"
        self.l1 = nn.Linear(in_features=input_shape, out_features=out, bias=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = self.sig(x)
        return x

    def get_model_name(self) -> str:
        return self.model_name
