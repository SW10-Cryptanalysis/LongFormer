 
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel
from transformers.modeling_outputs import CausalLMOutput
from config import Config


class LongformerForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.longformer = LongformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.longformer(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # Required for HuggingFace Trainer to save/load cleanly
    def save_pretrained(self, save_directory):
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Basic implementation to load config, then weights
        # (omitted full logic for brevity, assuming standard usage here)
        config = LongformerConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        # In a real scenario, you'd load state_dict here
        return model
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self):
        # Approximate
        return sum(p.numel() * p.element_size() for p in self.parameters())


def get_model():
    """Init model with params from config"""

    conf = LongformerConfig(
        vocab_size=Config.vocab_size,
        max_position_embeddings=Config.max_context,
        hidden_size=Config.dims,
        num_hidden_layers=Config.layers,
        intermediate_size=Config.dims * 4,
        num_attention_heads=Config.att_heads,
        attention_window=[512] * Config.layers,
        # Standard stuff
        hidden_act="silu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
    )

    model = LongformerForCausalLM(conf)
    print("Longformer Model loaded!")

    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model