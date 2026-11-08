"""Tiny CUDA contract checks executed within the authorized pilot reservation."""
import torch
from connector import AcousticMemory


def run():
    assert torch.cuda.is_available(), "No CPU fallback"
    torch.manual_seed(123)
    model = AcousticMemory(audio_dim=8, language_dim=12, width=6).cuda()
    audio = torch.randn(1, 8, 3, 8, device="cuda", requires_grad=True)
    question = torch.randn(1, 3, 12, device="cuda", requires_grad=True)
    fm = torch.ones(1, 8, 3, dtype=torch.bool, device="cuda")
    fm[:, 6:] = False
    sm = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    qm = torch.tensor([[True, True, False]], device="cuda")
    u = torch.randn(1, 8, 4, device="cuda")
    available = torch.ones(1, 8, dtype=torch.bool, device="cuda")
    args = dict(audio=audio, frame_mask=fm, span_mask=sm, question=question,
        question_mask=qm, uncertainty=u, uncertainty_available=available)
    saved = []
    def capture(module, inputs, output):
        output.retain_grad()
        saved.append(output)
    hook = model.score.register_forward_hook(capture)
    model.train()
    memory, chosen, _ = model(**args)
    memory[0].square().sum().backward()
    grad = saved[0].grad[0, :, 0]
    omitted = torch.ones(8, dtype=torch.bool, device="cuda")
    omitted[chosen[0]] = False
    valid = fm.any(-1)[0]
    assert (grad[omitted & valid].abs() > 0).all()
    assert (grad[~valid] == 0).all()
    assert (audio.grad[~fm] == 0).all()
    assert (question.grad[~qm] == 0).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    hook.remove()
    model.eval()
    with torch.no_grad():
        inference = model(**args)[0][0]
        assert torch.equal(memory[0], inference), "Train/eval hard forward mismatch"
        args["uncertainty_available"] = torch.zeros_like(available)
        args["uncertainty"] = torch.full_like(u, float("nan"))
        assert torch.isfinite(model(**args)[0][0]).all()
        uniform = model(**args, condition="uniform")
        assert uniform[1][0].tolist() == [0, 2, 3, 5]
        assert torch.isfinite(uniform[0][0]).all()
        args["frame_mask"] = torch.zeros_like(fm)
        assert model(**args)[0][0].shape == (0, 12)
        model.slots = 0
        args["frame_mask"] = fm
        assert model(**args)[0][0].shape == (0, 12)
        args["audio"] = audio[:, :0]
        args["frame_mask"] = fm[:, :0]
        args["span_mask"] = sm[:, :0]
        args["uncertainty"] = u[:, :0]
        args["uncertainty_available"] = available[:, :0]
        model.slots = 4
        assert model(**args)[0][0].shape == (0, 12)
    return {"uniform_indices_and_missing_posterior": True, "zero_candidate_spans_empty": True, "status": "pass", "device": torch.cuda.get_device_name(),
        "omitted_valid_logit_gradients": grad[omitted & valid].detach().tolist(),
        "invalid_logit_gradients": grad[~valid].detach().tolist(),
        "hard_train_eval_exact": True, "padding_gradients_zero": True,
        "missing_posterior_finite": True, "all_padding_empty": True, "k0_empty": True}
