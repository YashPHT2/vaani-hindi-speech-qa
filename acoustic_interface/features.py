"""Frozen predicted-token timestamp features, with no reference/answer arguments."""
import math
import time
import torch
from transformers.modeling_outputs import BaseModelOutput


@torch.no_grad()
def extract(whisper, processor, waveform, sample_rate=16000, block_tokens=4):
    if sample_rate != 16000 or len(waveform) / sample_rate > 30:
        raise ValueError("Prototype requires 16kHz audio of at most 30 seconds")
    timer_start = time.monotonic()
    device = next(whisper.parameters()).device
    batch = processor(waveform, sampling_rate=sample_rate, return_tensors="pt",
                      return_attention_mask=True)
    features = batch.input_features.to(device, dtype=next(whisper.parameters()).dtype)
    # Whisper expects the full padded 30-second feature window even for short clips.
    assert features.shape[-1] == 3000
    torch.cuda.synchronize()
    prepared = time.monotonic()
    encoder = whisper.model.encoder(features, return_dict=True)
    torch.cuda.synchronize()
    encoded = time.monotonic()
    out = whisper.generate(encoder_outputs=BaseModelOutput(last_hidden_state=encoder.last_hidden_state),
        attention_mask=batch.attention_mask.to(device), language="hi", task="transcribe",
        return_token_timestamps=True, return_dict_in_generate=True,
        output_logits=True, do_sample=False, max_new_tokens=224)
    torch.cuda.synchronize()
    decoded = time.monotonic()
    sequence = out["sequences"][0]
    timestamps = out["token_timestamps"][0]
    raw_logits = out["logits"]
    offset = len(sequence) - len(raw_logits)
    special = set(processor.tokenizer.all_special_ids)
    lexical = [i for i, tok in enumerate(sequence.tolist()) if tok not in special and i >= offset]
    transcript = processor.tokenizer.decode(sequence, skip_special_tokens=True).strip()
    assert processor.tokenizer.decode(sequence[lexical], skip_special_tokens=True).strip() == transcript
    duration = len(waveform) / sample_rate
    valid_frames = min(encoder.last_hidden_state.shape[1], math.ceil(duration / .02))
    assert timestamps.shape[0] == len(sequence)
    spans, audit = [], []
    for start in range(0, len(lexical), block_tokens):
        members = lexical[start:start + block_tokens]
        first, last = members[0], members[-1]
        t0 = max(0., min(duration, float(timestamps[first])))
        t1 = min(duration, float(timestamps[last + 1]) if last + 1 < len(timestamps) else duration)
        lo, hi = max(0, int(t0 / .02)), min(valid_frames, math.ceil(t1 / .02))
        if hi <= lo:
            audit.append({"positions": members, "discarded": "empty_timestamp_window"})
            continue
        stats = []
        for position in members:
            logits = raw_logits[position - offset][0].float()
            if not torch.isfinite(logits).all():
                raise RuntimeError("Need raw pre-constraint logits; encountered nonfinite vocabulary logits")
            logp = logits.log_softmax(-1)
            p = logp.exp()
            top = p.topk(5).values
            stats.append(torch.stack([-(p * logp).sum() / math.log(p.numel()),
                                      top[0] - top[1], logp[sequence[position]], top.sum()]))
        spans.append((encoder.last_hidden_state[0, lo:hi].float().cpu(), torch.stack(stats).mean(0).cpu()))
        audit.append({"positions": members, "token_ids": sequence[members].tolist(),
                      "text": processor.tokenizer.decode(sequence[members]), "start": t0, "end": t1,
                      "frame_start": lo, "frame_end": hi})
    # An unalignable prediction retains its text scaffold and receives K=0.
    # Never fabricate temporal evidence or silently drop the labeled view.
    max_frames = max((len(s[0]) for s in spans), default=1)
    audio = torch.zeros(1, len(spans), max_frames, encoder.last_hidden_state.shape[-1])
    frame_mask = torch.zeros(1, len(spans), max_frames, dtype=torch.bool)
    for s, (frames, _) in enumerate(spans):
        audio[0, s, :len(frames)] = frames
        frame_mask[0, s, :len(frames)] = True
    return {"audio": audio, "frame_mask": frame_mask,
            "span_mask": torch.ones(1, len(spans), dtype=torch.bool),
            "uncertainty": torch.stack([s[1] for s in spans])[None] if spans else torch.zeros(1, 0, 4),
            "uncertainty_available": torch.ones(1, len(spans), dtype=torch.bool),
            "transcript": transcript, "span_audit": audit,
            "alignment_available": bool(spans), "unalignable_reason": None if spans else "no_nonempty_predicted_token_spans",
            "posterior_definition": "raw_full_vocab_pre_constraint_logits_top5_mass",
            "alignment": "predicted_token_cross_attention_timestamps_blocks4",
            "feature_timings_s": {"processor_and_transfer": prepared-timer_start, "encoder": encoded-prepared,
                "decoder_and_dtw_alignment": decoded-encoded, "span_posterior_build": time.monotonic()-decoded}}
