import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class MultiTaskClassifier(nn.Module):
    """Transformer encoder with one classification head per task."""

    def __init__(self, model_name: str, load_pretrained: bool = True) -> None:
        """`load_pretrained=False` builds the encoder from its config only, so a
        fine-tuned checkpoint can be restored without downloading base weights."""
        super().__init__()
        if load_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleDict(
            {
                "department": nn.Linear(hidden_size, 5),
                "urgency": nn.Linear(hidden_size, 4),
                "sentiment": nn.Linear(hidden_size, 3),
            }
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = self.dropout(outputs.last_hidden_state[:, 0])
        return {name: head(pooled) for name, head in self.heads.items()}
