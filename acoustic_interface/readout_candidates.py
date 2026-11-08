"""Reference-free integer-frame candidate packing; no models or text inputs."""
import math


def grid_windows(valid_frames, step=20):
    assert type(valid_frames) is int and valid_frames > 0
    assert type(step) is int and step == 20
    return [(lo, min(valid_frames, lo + step)) for lo in range(0, valid_frames, step)]


def reported_windows(valid_frames, bounds=(145, 205)):
    lo, hi = bounds
    assert (lo, hi) == (145, 205) and hi <= valid_frames
    assert (hi - lo) % 4 == 0
    step = (hi - lo) // 4
    return [(start, start + step) for start in range(lo, hi, step)]


def coverage(windows, valid_frames):
    assert type(valid_frames) is int and valid_frames > 0
    union = set()
    total = 0
    for lo, hi in windows:
        assert type(lo) is int and type(hi) is int and 0 <= lo < hi <= valid_frames
        union.update(range(lo, hi))
        total += hi - lo
    return {"candidate_count": len(windows), "physical_union_frames": len(union),
        "physical_union_s": len(union) * .02, "coverage_fraction": len(union) / valid_frames,
        "valid_frames": valid_frames, "summed_window_frames": total, "overlap_frames": total - len(union)}


def pack_windows(encoder_hidden, valid_frames, windows):
    import torch
    assert encoder_hidden.ndim == 2 and encoder_hidden.shape[-1] == 768
    assert 0 < valid_frames <= len(encoder_hidden)
    windows = list(windows)
    assert all(type(lo) is int and type(hi) is int for lo, hi in windows)
    stats = coverage(windows, valid_frames)
    width = max((hi - lo for lo, hi in windows), default=1)
    audio = torch.zeros(1, len(windows), width, 768, dtype=torch.float32, device=encoder_hidden.device)
    mask = torch.zeros(1, len(windows), width, dtype=torch.bool, device=encoder_hidden.device)
    for index, (lo, hi) in enumerate(windows):
        audio[0, index, :hi-lo] = encoder_hidden[lo:hi].float()
        mask[0, index, :hi-lo] = True
    return {"audio": audio, "frame_mask": mask,
        "span_mask": torch.ones(1, len(windows), dtype=torch.bool, device=encoder_hidden.device),
        "uncertainty": torch.zeros(1, len(windows), 4, device=encoder_hidden.device),
        "uncertainty_available": torch.zeros(1, len(windows), dtype=torch.bool, device=encoder_hidden.device),
        "candidate_windows": windows, "coverage": stats}


def cached_windows(features, valid_frames):
    import torch
    valid = features["span_mask"][0].bool() & features["frame_mask"].any(-1)[0]
    audit = [row for row in features["span_audit"] if "discarded" not in row]
    assert len(audit) == features["audio"].shape[1]
    windows = [(int(row["frame_start"]), int(row["frame_end"])) for row in audit]
    selected = [windows[i] for i in range(len(windows)) if bool(valid[i])]
    stats = coverage(selected, valid_frames)
    return windows, stats


def replace_geometry_audio(features, donor_hidden, valid_frames):
    """Donor frames in identical recipient windows; preserve U and routing masks."""
    import torch
    windows, _ = cached_windows(features, valid_frames)
    result = {key: features[key].clone() for key in
        ("audio", "frame_mask", "span_mask", "uncertainty", "uncertainty_available")}
    result["audio"].zero_()
    for index, (lo, hi) in enumerate(windows):
        count = int(features["frame_mask"][0, index].sum())
        assert count == hi-lo and hi <= len(donor_hidden)
        result["audio"][0, index, features["frame_mask"][0, index]] = donor_hidden[lo:hi].to(result["audio"])
    for key in ("frame_mask", "span_mask", "uncertainty", "uncertainty_available"):
        assert torch.equal(result[key], features[key])
    return result
