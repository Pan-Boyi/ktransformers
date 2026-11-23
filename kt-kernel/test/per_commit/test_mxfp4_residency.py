# SPDX-License-Identifier: Apache-2.0
"""Check AVX2 MXFP4 full-set loading and expert demotion in a CPU-only process."""

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def without_pinned_memory(factory):
    def allocate(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return factory(*args, **kwargs)

    return allocate


def run(threads, nodes):
    os.environ.update(
        TORCH_DEVICE_BACKEND_AUTOLOAD="0",
        ASCEND_RT_VISIBLE_DEVICES="",
        KT_FORCE_SYNC_SUBMIT="1",
        KT_EXTERNAL_NPU_REPORT_SUBSCRIBER="1",
        KT_MXFP4_BACKEND="avx2",
        KT_MXFP4_COMPACT_SCALES="0",
        KT_ZEROCOPY_WEIGHTS="0",
    )

    import torch
    from safetensors.torch import save_file

    from kt_kernel import KTMoEWrapper, experts_base, kt_kernel_ext

    width = 128
    resident_id = 1
    projections = ("w1", "w3", "w2")
    rng = torch.Generator().manual_seed(9381)
    tensors = {}
    for expert in range(2):
        for proj in projections:
            key = f"layers.0.ffn.experts.{expert}.{proj}"
            tensors[f"{key}.weight"] = torch.randint(0, 256, (width, width // 2), dtype=torch.uint8, generator=rng)
            tensors[f"{key}.scale"] = torch.randint(121, 127, (width, width // 32), dtype=torch.uint8, generator=rng)

    def load(directory, mask, fullset):
        with patch.dict(os.environ, KT_FULLSET_LOAD="1" if fullset else "0"):
            wrapper = KTMoEWrapper(
                layer_idx=0,
                num_experts=2,
                num_experts_per_tok=1,
                hidden_size=width,
                moe_intermediate_size=width,
                gpu_experts_mask=torch.tensor(mask),
                cpuinfer_threads=threads,
                threadpool_count=len(nodes),
                weight_path=directory,
                chunked_prefill_size=32,
                method="MXFP4",
                numa_nodes=nodes,
                swiglu_limit=10.0,
            )
            wrapper.load_weights(torch.arange(2, dtype=torch.int64))
        assert isinstance(wrapper.moe, kt_kernel_ext.moe.AVX2MXFP4_MOE)
        return wrapper

    hidden = torch.randn((1, width), generator=rng).bfloat16()
    ids, routing = torch.tensor([[resident_id]]), torch.ones((1, 1))
    allocators = {name: without_pinned_memory(getattr(torch, name)) for name in ("empty", "zeros", "full")}
    with (
        TemporaryDirectory(prefix="kt-mxfp4-residency-") as directory,
        patch.multiple(torch, **allocators),
        patch.object(experts_base, "_ensure_ascend_callback_worker", return_value=None),
    ):
        save_file(tensors, str(Path(directory) / "model.safetensors"))

        # Use the ordinary all-CPU load as the control for the same AVX2 kernel.
        expected = load(directory, [False, False], False).forward(hidden, ids, routing, 0).clone()
        assert torch.isfinite(expected).all() and expected.count_nonzero().item() > 0

        wrapper = load(directory, [False, True], True)
        assert wrapper.cpu_weight_expert_ids() == [0, 1]
        weights = torch.empty((3, width, width // 2), dtype=torch.uint8)
        scales = torch.empty((3, width, width // 32), dtype=torch.bfloat16)
        wrapper.submit_write_weight_scale_to_buffer(
            1, resident_id, [weights.data_ptr()], [scales.data_ptr()], [weights[2].data_ptr()], [scales[2].data_ptr()]
        )
        wrapper.sync_write_weight_scale_to_buffer()
        for i, proj in enumerate(projections):
            key = f"layers.0.ffn.experts.{resident_id}.{proj}"
            assert torch.equal(weights[i], tensors[f"{key}.weight"]), f"{proj} weight readback"
            scale_bits = tensors[f"{key}.scale"].to(torch.int16) << 7
            assert torch.equal(scales[i].view(torch.int16), scale_bits), f"{proj} scale readback"

        assert torch.count_nonzero(wrapper.forward(hidden, ids, routing, 0)).item() == 0
        wrapper.gpu_experts_mask[resident_id] = False
        torch.testing.assert_close(wrapper.forward(hidden, ids, routing, 0), expected, rtol=0, atol=0)

    print("VERDICT: PASS (AVX2 full-set loading and CPU demotion)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nodes", type=lambda value: [int(node) for node in value.split(",")], default="0")
    args = parser.parse_args()
    run(args.threads, args.nodes)
