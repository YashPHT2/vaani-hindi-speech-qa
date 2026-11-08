"""Question-conditioned, fixed-budget acoustic soft memory (no oracle inputs)."""
import math
import torch
from torch import nn


class AcousticMemory(nn.Module):
    def __init__(self, audio_dim=768, language_dim=1536, width=64, slots=4):
        super().__init__()
        if not 0 <= slots <= 4:
            raise ValueError("Fixed slot budget must be 0..4")
        self.slots = slots
        self.audio_dim, self.language_dim = audio_dim, language_dim
        self.query = nn.Linear(language_dim, width)
        self.key = nn.Linear(audio_dim, width)
        self.value = nn.Linear(audio_dim, width)
        self.score = nn.Sequential(nn.Linear(width * 2 + 5, width), nn.GELU(), nn.Linear(width, 1))
        self.project = nn.Linear(width, language_dim)
        self.register_buffer("uncertainty_mean", torch.zeros(4))
        self.register_buffer("uncertainty_std", torch.ones(4))

    def forward(self, audio, frame_mask, span_mask, question, question_mask,
                uncertainty, uncertainty_available, condition="full"):
        """Return one variable-length memory per example, indices and selected gates.

        Training uses a biased straight-through localized soft router when S>K.
        Forward values remain exactly the chronological hard top-k memory.
        """
        assert audio.ndim == 4 and audio.shape[-1] == self.audio_dim
        assert question.shape[-1] == self.language_dim
        assert frame_mask.shape == audio.shape[:3]
        assert span_mask.shape == audio.shape[:2]
        assert uncertainty.shape == (*audio.shape[:2], 4)
        valid = span_mask.bool() & frame_mask.any(-1)
        frame_mask = frame_mask.bool() & valid[..., None]
        audio = audio.masked_fill(~frame_mask[..., None], 0)
        question = question.masked_fill(~question_mask.bool()[..., None], 0)
        qmask = question_mask.to(question.dtype).unsqueeze(-1)
        pooled = (question * qmask).sum(1) / qmask.sum(1).clamp_min(1)
        query = self.query(pooled)
        if condition == "query_off":
            query = query * 0
        if condition == "acoustic_off":
            audio = audio * 0
        keys, values = self.key(audio), self.value(audio)
        scores = (keys * query[:, None, None, :]).sum(-1) / math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~frame_mask.bool(), -1e4)
        attention = scores.softmax(-1) * frame_mask
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled_audio = (attention[..., None] * values).sum(-2)
        u = ((uncertainty - self.uncertainty_mean) / self.uncertainty_std.clamp_min(1e-5))
        available = uncertainty_available.bool() & valid
        if condition in ("uncertainty_off", "uniform"):
            available = torch.zeros_like(available)
        u = u.masked_fill(~available[..., None], 0)
        logits = self.score(torch.cat([pooled_audio, query[:, None].expand_as(pooled_audio),
                                      u, available[..., None].to(u.dtype)], -1)).squeeze(-1)
        memories, selections, gates = [], [], []
        for b in range(audio.shape[0]):
            count = min(self.slots, int(valid[b].sum()))
            # Stable descending sort breaks ties by original chronological index.
            chosen = torch.argsort(logits[b].masked_fill(~valid[b], -torch.inf),
                                   descending=True, stable=True)[:count].sort().values
            if condition == "uniform" and count:
                candidates = valid[b].nonzero(as_tuple=True)[0]
                positions = torch.linspace(0, len(candidates) - 1, count, device=audio.device).round().long()
                chosen = candidates[positions]
            gate = logits[b, chosen].sigmoid()
            content = self.project(pooled_audio[b]) * logits[b].sigmoid()[:, None]
            hard = content[chosen]
            if self.training and condition != "uniform" and count and int(valid[b].sum()) > count:
                # Normalized rank among valid spans, including non-contiguous masks.
                rank = (valid[b].cumsum(0) - 1).to(logits.dtype)
                times = rank / (valid[b].sum() - 1).clamp_min(1)
                anchors = times[chosen].detach()
                local = logits[b][None] - 2 * (times[None] - anchors[:, None]).abs()
                relaxed = local.masked_fill(~valid[b][None], -torch.inf).softmax(-1)
                hard_selector = torch.zeros_like(relaxed)
                hard_selector.scatter_(1, chosen[:, None], 1)
                # Subtraction before addition makes the hard forward exact.
                selector = hard_selector + (relaxed - relaxed.detach())
                hard = selector @ content
            memories.append(hard)
            selections.append(chosen)
            gates.append(gate)
        return memories, selections, gates


def pack_prompt(lm, tokenizer, transcript, question, memory, answer=None):
    """Single-example packing: K=0 literally inserts zero acoustic positions."""
    before = "<|im_start|>system\nकेवल दिए गए साक्ष्य से छोटा उत्तर दें। जानकारी न हो तो अज्ञात लिखें।<|im_end|>\n<|im_start|>user\nप्रतिलेख: " + transcript + "\nध्वनिक साक्ष्य:\n"
    after = "\nप्रश्न: " + question + "<|im_end|>\n<|im_start|>assistant\n"
    device = lm.get_input_embeddings().weight.device
    def embed(text):
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        return ids, lm.get_input_embeddings()(ids)[0]
    _, a = embed(before)
    _, z = embed(after)
    pieces = [a, memory.to(a.dtype), z]
    prompt_length = sum(len(p) for p in pieces)
    if answer is not None:
        target_ids, target = embed(answer + "<|im_end|>")
        pieces.append(target)
    inputs = torch.cat(pieces)[None]
    mask = torch.ones(inputs.shape[:2], dtype=torch.long, device=device)
    labels = None
    if answer is not None:
        labels = torch.full(inputs.shape[:2], -100, dtype=torch.long, device=device)
        labels[:, prompt_length:] = target_ids
    return dict(inputs_embeds=inputs, attention_mask=mask,
                position_ids=torch.arange(inputs.shape[1], device=device)[None], labels=labels)
