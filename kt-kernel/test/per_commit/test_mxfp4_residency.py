# SPDX-License-Identifier: Apache-2.0
"""Native MXFP4 residency regression. CPU only; no checkpoint or NPU needed.

Run as a fresh process for each topology, for example:
  PYTHONPATH=<site> python test_mxfp4_residency.py --threads 4 --nodes 0,1
No missing-extension skip: inability to load/compute is a failure.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["KT_FORCE_SYNC_SUBMIT"] = "1"
os.environ["KT_EXTERNAL_NPU_REPORT_SUBSCRIBER"] = "1"

import torch
from safetensors.torch import save_file

from kt_kernel import KTMoEWrapper, __cpu_variant__, kt_kernel_ext
import kt_kernel.experts_base as experts_base
from kt_kernel.utils.amx import NativeMoEWrapper, _select_mxfp4_backend, mxfp4_backend_capabilities

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 128, 128, 8, 2
LIMIT = 10.0
LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6])


def decode(weight, scale):
    codes = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    return LUT[codes.long()] * torch.exp2(scale.float() - 127).repeat_interleave(32, dim=1)


def bf(value):
    return value.bfloat16().float()


def reference(x, ids, routing, weights, mask, *, emulate=False, acc64=False, partitions=1):
    out = torch.zeros(x.shape[0], HIDDEN)
    for expert in range(EXPERTS):
        if mask[expert]:
            continue
        tok, pos = torch.where(ids == expert)
        if not tok.numel():
            continue
        w1, w3, w2 = weights[expert]
        a = x[tok].float()
        if acc64:
            a, w1, w3, w2 = a.double(), w1.double(), w3.double(), w2.double()
        gate, up = (a @ w1.T).float(), (a @ w3.T).float()
        if emulate:
            gate, up = bf(gate), bf(up)
        hidden = torch.nn.functional.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        if emulate:
            hidden = bf(hidden)
        if emulate:
            # Each TP partition stores its partial down result as BF16.
            # The weighted partials are merged in FP32 before the final BF16
            # store. Keep the fp32 mathematical reference unpartitioned.
            y = torch.zeros(hidden.shape[0], HIDDEN)
            for h_part, w_part in zip(hidden.chunk(partitions, dim=1), w2.chunk(partitions, dim=1)):
                partial = (h_part.double() @ w_part.T).float() if acc64 else h_part @ w_part.T
                y += bf(partial)
        else:
            y = (hidden.double() @ w2.T).float() if acc64 else hidden @ w2.T
        out.index_add_(0, tok, routing[tok, pos, None] * y)
    return bf(out) if emulate else out


def rel_l2(got, expected):
    return float((got.double() - expected.double()).norm() / expected.double().norm().clamp_min(1e-30))


def export(wrapper, expert):
    w13 = torch.empty(2 * INTERMEDIATE * HIDDEN // 2, dtype=torch.uint8)
    s13 = torch.empty(2 * INTERMEDIATE * HIDDEN // 32, dtype=torch.bfloat16)
    w2 = torch.empty(HIDDEN * INTERMEDIATE // 2, dtype=torch.uint8)
    s2 = torch.empty(HIDDEN * INTERMEDIATE // 32, dtype=torch.bfloat16)
    wrapper.submit_write_weight_scale_to_buffer(
        1, expert, [w13.data_ptr()], [s13.data_ptr()], [w2.data_ptr()], [s2.data_ptr()])
    wrapper.sync_write_weight_scale_to_buffer()
    scales = torch.cat((s13, s2)).view(torch.int16).to(torch.int32)
    assert not (scales & 0x807F).any(), "non-E8M0 bits in exported scales"
    return torch.cat((w13, w2)), ((scales >> 7) & 255).byte()


def check_flat_fullset(raw, mask, cpu):
    """The public flat-buffer API must also fill initially masked experts."""
    config = kt_kernel_ext.moe.MOEConfig(EXPERTS, TOPK, HIDDEN, INTERMEDIATE, mask.data_ptr())
    config.pool, config.max_len = cpu.backend_, 32
    config.quant_config.bits, config.quant_config.group_size = 4, 32
    config.quant_config.zero_point = False
    config.skip_gpu_expert_weights = False
    held = []
    for index, projection in enumerate(("gate", "up", "down")):
        weight = torch.stack([raw[e][0][index] for e in range(EXPERTS)]).contiguous()
        scale_bytes = torch.stack([raw[e][1][index] for e in range(EXPERTS)])
        scale = (scale_bytes.int() << 7).short().view(torch.bfloat16).contiguous()
        held.extend((weight, scale))
        setattr(config, projection + "_proj", weight.data_ptr())
        setattr(config, projection + "_scale", scale.data_ptr())
    native = _select_mxfp4_backend()(config)
    mapping = torch.arange(EXPERTS, dtype=torch.int64)
    cpu.submit(native.load_weights_task(mapping.data_ptr()))
    cpu.sync()
    adapter = SimpleNamespace(
        submit_write_weight_scale_to_buffer=lambda *args: cpu.submit(native.write_weight_scale_to_buffer_task(*args)),
        sync_write_weight_scale_to_buffer=cpu.sync)
    for expert in range(EXPERTS):
        weight, scale = export(adapter, expert)
        assert torch.equal(weight, torch.cat([v.flatten() for v in raw[expert][0]])), "flat weight placement"
        assert torch.equal(scale, torch.cat([v.flatten() for v in raw[expert][1]])), "flat scale placement"
    return EXPERTS


def run(threads, nodes, backend):
    os.environ["KT_MXFP4_BACKEND"] = backend
    os.environ["KT_MXFP4_COMPACT_SCALES"] = "0"
    torch.set_num_threads(2)
    # Pure CPU lockout: remove pinned allocation and ACL worker startup, as in
    # the real-checkpoint checker. No numerical operation is replaced.
    for name in ("empty", "zeros", "full"):
        original = getattr(torch, name)

        def unpinned(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        setattr(torch, name, unpinned)
    experts_base._ensure_ascend_callback_worker = lambda: None
    print(json.dumps({"variant": __cpu_variant__, "capabilities": mxfp4_backend_capabilities(),
                      "threads": threads, "nodes": nodes}), flush=True)
    generator = torch.Generator().manual_seed(9381)
    tensors, raw, decoded = {}, {}, {}
    for layer in (0, 1):
        raw[layer], decoded[layer] = {}, {}
        for expert in range(EXPERTS):
            packed, scales = [], []
            for projection, rows, cols in (("w1", INTERMEDIATE, HIDDEN),
                                            ("w3", INTERMEDIATE, HIDDEN),
                                            ("w2", HIDDEN, INTERMEDIATE)):
                w = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator)
                s = torch.randint(121, 127, (rows, cols // 32), dtype=torch.uint8, generator=generator)
                key = f"layers.{layer}.ffn.experts.{expert}.{projection}"
                tensors[key + ".weight"], tensors[key + ".scale"] = w, s
                packed.append(w)
                scales.append(s)
            raw[layer][expert] = (packed, scales)
            decoded[layer][expert] = tuple(decode(w, s) for w, s in zip(packed, scales))
    checks = 0
    with tempfile.TemporaryDirectory(prefix="kt-mxfp4-residency-") as directory:
        save_file(tensors, str(Path(directory) / "model.safetensors"))
        for fullset in (False, True):
            os.environ["KT_FULLSET_LOAD"] = "1" if fullset else "0"
            for layer in (0, 1):
                mask = torch.zeros(EXPERTS, dtype=torch.bool)
                mask[[1, 3]] = True
                wrapper = KTMoEWrapper(
                    layer_idx=layer, num_experts=EXPERTS, num_experts_per_tok=TOPK,
                    hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE,
                    gpu_experts_mask=mask, cpuinfer_threads=threads, threadpool_count=len(nodes),
                    weight_path=directory, chunked_prefill_size=32, method="MXFP4",
                    numa_nodes=nodes, swiglu_limit=LIMIT)
                wrapper.load_weights(torch.arange(EXPERTS, dtype=torch.int64))
                loaded = list(range(EXPERTS)) if fullset else [0, 2, 4, 5, 6, 7]
                assert wrapper.cpu_weight_expert_ids() == loaded
                for expert in loaded:
                    w, s = export(wrapper, expert)
                    packed, scales = raw[layer][expert]
                    assert torch.equal(w, torch.cat([v.flatten() for v in packed])), "weight placement/order"
                    expected = torch.cat([v.flatten() for v in scales])
                    assert torch.equal(s, expected), "E8M0 byte roundtrip"
                    mutated = expected.clone()
                    mutated[9] ^= 1
                    assert (s != mutated).sum().item() == 1, "scale negative control"
                    checks += 1
                if not fullset:
                    try:
                        export(wrapper, 1)
                    except RuntimeError as exc:
                        assert "no loaded CPU weights" in str(exc)
                    else:
                        raise AssertionError("subset export read an unloaded expert")
                # No reloads: demote an initially resident expert, promote a CPU
                # expert, then demote all and restore the initial placement.
                stages = [[1, 3], [2, 3], [], [1, 3]] if fullset else [[1, 3]]
                for resident in stages:
                    wrapper.gpu_experts_mask.zero_()
                    wrapper.gpu_experts_mask[resident] = True
                    assert wrapper.cpu_weight_expert_ids() == loaded, "loaded slots followed live routing mask"
                    for tokens in (1, 4, 32):
                        ids = torch.stack([torch.randperm(EXPERTS, generator=generator)[:TOPK] for _ in range(tokens)])
                        ids[0] = torch.tensor([0, 1])
                        if tokens >= 4:
                            ids[1] = torch.tensor([1, 3])
                        score = torch.nn.functional.softplus(torch.randn(tokens, TOPK, generator=generator)).sqrt()
                        routing = (score / score.sum(-1, keepdim=True) * 1.5).contiguous()
                        x = (torch.randn(tokens, HIDDEN, generator=generator) * 8).bfloat16()
                        got = wrapper.forward(x, ids, routing, 0).float().clone()
                        ref = reference(x, ids, routing, decoded[layer], wrapper.gpu_experts_mask)
                        emu = reference(x, ids, routing, decoded[layer], wrapper.gpu_experts_mask,
                                        emulate=True, partitions=len(nodes))
                        emu64 = reference(x, ids, routing, decoded[layer], wrapper.gpu_experts_mask,
                                          emulate=True, acc64=True, partitions=len(nodes))
                        tau_a = 2 * max(rel_l2(emu, ref), 2**-8)
                        tau_b = 2 * max(rel_l2(emu, emu64), 2**-12)
                        assert torch.isfinite(got).all()
                        assert rel_l2(got, ref) <= tau_a, ("fp32", layer, tokens, resident, rel_l2(got, ref), tau_a)
                        assert rel_l2(got, emu) <= tau_b, ("bf16", layer, tokens, resident, rel_l2(got, emu), tau_b)
                        assert rel_l2(torch.zeros_like(got), ref) > tau_a, "missing-expert negative control"
                        assert rel_l2(got, ref * 2) > tau_a, "whole-scale negative control"
                        checks += 1
                if fullset:
                    checks += check_flat_fullset(raw[layer], mask, wrapper.cpu_infer)
                del wrapper
                NativeMoEWrapper.force_release_loader()
    assert "torch_npu" not in __import__("sys").modules
    print(f"VERDICT: PASS ({checks} native residency/roundtrip/numeric cases)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--nodes", default="0")
    parser.add_argument("--backend", choices=("avx2", "amx"), default="avx2")
    args = parser.parse_args()
    run(args.threads, [int(node) for node in args.nodes.split(",")], args.backend)
