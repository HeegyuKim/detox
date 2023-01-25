import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Config, CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple, Union


class DirectorHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.cls_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        last_hidden_states: torch.Tensor,
        labels: torch.Tensor,
        class_labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids: (bs, seq, hidden)
        attention_masks: (bs, seq)
        class_labels: (bs, )
        """

        out_cls_logits = self.cls_head(last_hidden_states).sigmoid()  # (bs, seq, vocab)
        loss = None

        if class_labels is not None:
            cls_logits = out_cls_logits[:, -1, :]
            labels = cls_logits[:, -1]
            label_mask = attention_mask[:, -1]

            cls_logits = cls_logits.gather(labels.unsqueeze(-1))  # (bs, seq)
            class_prob = class_labels.repeat(-1, cls_logits.shape[1])  # (bs, seq)
            loss = F.binary_cross_entropy(class_prob, class_prob, reduction="none")
            loss = loss.masked_fill(label_mask == 0, 0)
            loss = loss.sum() / label_mask.sum()

        return out_cls_logits, loss


class DirectorModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)

        self.director_head = DirectorHead(config.hidden_size, config.vocab_size)

    def freeze_gpt(self):
        for p in self.parameters():
            p.requires_grad = False
            
        for p in self.director_head.parameters():
            p.requires_grad = True

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        class_labels: Optional[torch.LongTensor] = None,
        gamma: Optional[float] = None,
        generate_positive: bool = True
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        output_hidden_states = True
        
        out = super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

        cls_logits, cls_loss = self.director_head(
            last_hidden_states=out.hidden_states[:, -1, :],
            labels=labels,
            class_labels=class_labels,
            attention_mask=attention_mask,
        )

        if not generate_positive:
            cls_logits = 1 - cls_logits
        if gamma is not None:
            cls_logits = cls_logits.pow(gamma)

        out.logits = (out.logits.softmax() + cls_logits).softmax()

        if out.loss is not None:
            out.loss += cls_loss    

        return out
