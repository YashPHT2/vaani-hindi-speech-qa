"""Answer-token alignment for frozen-decoder distillation."""


def answer_targets(labels):
    """Return target IDs in decoder prediction order, including assistant EOS."""
    assert labels.ndim == 2 and labels.shape[0] == 1
    shifted = labels[0, 1:]
    mask = shifted.ne(-100)
    assert mask.any() and not mask[0]
    first = int(mask.nonzero(as_tuple=True)[0][0])
    assert mask[first:].all(), "Answer labels must form one contiguous suffix"
    return shifted[mask], mask


def answer_distribution(logits, labels):
    assert logits.ndim == 3 and logits.shape[:2] == labels.shape
    targets, mask = answer_targets(labels)
    return logits[0, :-1][mask], targets


def matched_loss(student_logits, student_ids, teacher_logits=None, teacher_ids=None):
    """Per-view CE plus the temperature-corrected forward KL when eligible."""
    import torch
    from torch.nn import functional as F

    assert student_logits.ndim == 2 and student_ids.shape == student_logits.shape[:1]
    student = student_logits.float()
    ce = F.cross_entropy(student, student_ids)
    if teacher_logits is None:
        assert teacher_ids is None
        return ce, ce, ce.new_zeros(())
    assert teacher_logits.shape == student_logits.shape and torch.equal(teacher_ids, student_ids)
    assert not teacher_logits.requires_grad
    student_logp = F.log_softmax(student / 2, dim=-1)
    teacher_logp = F.log_softmax(teacher_logits.float() / 2, dim=-1)
    kl = F.kl_div(student_logp, teacher_logp, log_target=True, reduction="batchmean")
    assert torch.isfinite(kl) and kl >= -1e-5
    return ce + .5 * (2 ** 2) * kl, ce, kl
