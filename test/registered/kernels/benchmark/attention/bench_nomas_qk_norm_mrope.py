# SPDX-License-Identifier: Apache-2.0
"""Compare the QK norm/MRoPE tail with supplied ATen reductions (cold L2)."""

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.attention import nomas_qk_norm_mrope
from sglang.multimodal_gen.runtime.models.vaes import nomas as nm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def previous(x, mt, mh, wt, wh, cos, sin, eps, out):
    # Include the temporary-result copy removed by the direct-output kernel.
    out.copy_(
        nm._compiled_qk_norm_mrope()(
            x, mt, mh, eps, wt, wh, cos, sin, (64, 32, 32), x.dtype
        )
    )
    return out


@marker.parametrize("tokens", [8800, 35200])
@marker.parametrize("heads", [8, 32])
@marker.benchmark("impl", ["previous", "direct_output"])
def benchmark(tokens: int, heads: int, impl: str):
    import torch

    x = create_random(1, tokens, heads, 128)
    wt, wh = create_random(64), create_random(64)
    angles = create_random(1, tokens, 1, 128, dtype=torch.float32)
    mt, mh = nm._mean_square(x[..., :64]), nm._mean_square(x[..., 64:])
    out = torch.empty_like(x)
    fn = previous if impl == "previous" else nomas_qk_norm_mrope
    # Match the decoder's eager launch mode and flush L2 before each invocation.
    # The compiled reference may specialize on cloned inputs during graph capture.
    # ATen reductions are supplied to both implementations and excluded here.
    with torch.inference_mode():
        return marker.do_bench(
            fn,
            use_cuda_graph=False,
            input_args=(x, mt, mh, wt, wh, angles.cos(), angles.sin(), 1e-6, out),
        )


if __name__ == "__main__":
    benchmark.run()
