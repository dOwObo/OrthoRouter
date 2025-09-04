# nlp_task2vec/modeling_probe.py
import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class ProbeClassifier(nn.Module):
    def __init__(self, backbone_name="roberta-base", num_labels=None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.hidden = self.backbone.config.hidden_size
        for p in self.backbone.parameters():
            p.requires_grad_(False)   # 凍結 probe
        if num_labels is None:
            num_labels = 2
        self.classifier = nn.Linear(self.hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0]
        logits = self.classifier(h)
        return logits

def get_tokenizer(backbone_name="roberta-base"):
    return AutoTokenizer.from_pretrained(backbone_name, use_fast=True)