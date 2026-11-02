# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.attention import nomas_qk_norm_mrope
from sglang.multimodal_gen.runtime.models.vaes import nomas as nm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="CUDA required",
)


def inputs(batch, tokens, heads):
    torch.manual_seed(20261102)
    x = torch.randn(batch, tokens, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = [(1 + 0.2 * torch.randn(64, device="cuda")).bfloat16() for _ in range(2)]
    angles = torch.randn(1, tokens, 1, 128, device="cuda")
    return x, *weights, angles.cos(), angles.sin()


def reference(x, mt, mh, wt, wh, cos, sin, eps):
    # The compiled pre-change path, including its FP32 arithmetic/cast boundaries.
    return nm._compiled_qk_norm_mrope()(
        x, mt, mh, eps, wt, wh, cos, sin, (64, 32, 32), torch.bfloat16
    )


@pytest.mark.parametrize(
    "batch,tokens,heads",
    [(1, 1, 1), (1, 17, 8), (2, 257, 32), (1, 8800, 32), (1, 35200, 8)],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
@torch.inference_mode()
def test_matches_previous_path(batch, tokens, heads, eps):
    x, wt, wh, cos, sin = inputs(batch, tokens, heads)
    # Vary norms widely and include zero to exercise epsilon and BF16 rounding.
    x[:, ::7] *= 100
    x[:, ::11] *= 0.0001
    x[:, ::13] = 0
    mt, mh = nm._mean_square(x[..., :64]), nm._mean_square(x[..., 64:])
    expected = reference(x, mt, mh, wt, wh, cos, sin, eps)
    actual = nomas_qk_norm_mrope(x, mt, mh, wt, wh, cos, sin, eps)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_strided_chunks_write_only_output_slice():
    x, wt, wh, cos, sin = inputs(2, 43, 8)
    storage = torch.full_like(x, 17)
    # Slice both batch strides and token strides, as in a bounded decoder chunk.
    part, dst = x[:, 3:37:2], storage[:, 3:37:2]
    c, s = cos[:, 3:37:2], sin[:, 3:37:2]
    mt, mh = nm._mean_square(part[..., :64]), nm._mean_square(part[..., 64:])
    expected = reference(part, mt, mh, wt, wh, c, s, 1e-6)
    result = nomas_qk_norm_mrope(part, mt, mh, wt, wh, c, s, 1e-6, out=dst)
    assert result is dst
    expected_storage = torch.full_like(storage, 17)
    expected_storage[:, 3:37:2] = expected
    assert torch.equal(storage.view(torch.int16), expected_storage.view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_runtime_chunked_integration(monkeypatch, dtype):
    # This case adds dtype and forced-chunk specializations after the large shapes.
    torch.compiler.reset()
    x, wt, wh, cos, sin = inputs(2, 43, 8)
    x = x.to(dtype)
    arch = nm.GenExpertArch(hidden_size=128, num_heads=1, num_kv_heads=1)
    attn = nm.GenAttention(arch).cuda().bfloat16().eval()
    attn.q_norm_t.weight.copy_(wt)
    attn.q_norm_hw.weight.copy_(wh)
    # Force multiple chunks, including a short final chunk and nontrivial batch stride.
    monkeypatch.setattr(nm, "_QK_FUSED_CHUNK_BYTES", 2 * 8 * 128 * 4 * 7)
    expected = reference(
        x,
        nm._mean_square(x[..., :64]),
        nm._mean_square(x[..., 64:]),
        wt,
        wh,
        cos,
        sin,
        1e-6,
    )
    actual = attn._prepare_qk(
        x, cos, sin, attn.q_norm_t, attn.q_norm_hw, torch.bfloat16
    )
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@torch.inference_mode()
def test_rejects_unsupported_contract():
    x, wt, wh, cos, sin = inputs(1, 17, 8)
    mt, mh = nm._mean_square(x[..., :64]), nm._mean_square(x[..., 64:])
    with pytest.raises(ValueError, match="BF16"):
        nomas_qk_norm_mrope(x.float(), mt, mh, wt, wh, cos, sin, 1e-6)
    with pytest.raises(ValueError, match="means"):
        nomas_qk_norm_mrope(x, mt.squeeze(-1), mh, wt, wh, cos, sin, 1e-6)
    with pytest.raises(ValueError, match="RoPE"):
        nomas_qk_norm_mrope(x, mt, mh, wt, wh, cos.bfloat16(), sin, 1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
