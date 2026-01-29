import torch


def check_tol_diff(
    norm: torch.Tensor,
    ref: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
    ptol: float = 1.0,
    mixed_tol: bool = False,
):
    """
    Args:
        norm: The computed/estimated value to be validated.
        ref: The reference/ground truth value for comparison.
        atol: The absolute tolerance.
        rtol: The relative tolerance.
        ptol: The percentage tolerance. When match_ratio >= ptol is considered to pass.
        mixed_tol: If true, atol, rtol and ptol are ignored.
    """
    if isinstance(norm, tuple) or isinstance(norm, list):
        for norm_i, ref_i in zip(norm, ref):
            check_tol_diff(norm_i, ref_i, atol, rtol, ptol, mixed_tol)
        return
    
    if mixed_tol:
        mask = ref.abs() < 1.0
        tmpatol = tmprtol = 2**-6
        torch.testing.assert_close(norm[mask], ref[mask], atol=tmpatol, rtol=0)
        torch.testing.assert_close(norm[~mask], ref[~mask], atol=0, rtol=tmprtol)

    elif ptol != 1.0:
        assert ptol < 1.0, f"{ptol=} should <= 1.0"

        matches = torch.isclose(norm, ref, rtol=rtol, atol=atol)
        total = matches.numel()
        match = int(torch.sum(matches))
        mismatch = total - match
        match_ratio = match / total

        assert match_ratio >= ptol, f"{match_ratio=:.5%} ({match=} / {mismatch=} / {total=}) is under {ptol=:%}, Please Check!"

    else:
        torch.testing.assert_close(norm.to(torch.float32), ref.to(torch.float32), atol=atol, rtol=rtol)
