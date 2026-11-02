"""
Post-training int8 quantization of Compact n_layers=3, part by part
(encoder, translator, decoder, all, then each conv alone): test MSE and
SSIM, gap to the fp32 output, size of the saved weights, CPU latency on
one thread.

Static quantization in FX mode (prepare_fx / convert_fx), oneDNN
backend, calibrated on 200 sequences. FX can't trace the full forward,
so each part's nn.Sequential is quantized on its own. The decoder's
ConvTranspose2d layers stay in fp32 (see check_convtranspose_kernel).

Usage:
    python scripts/quantize_int8.py --data-root data --ckpt-root checkpoints
"""
import sys
import io
import copy
import json
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.ao.quantization import (
    get_default_qconfig_mapping, QConfigMapping, QConfig,
    HistogramObserver, default_weight_observer,
)
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

from src.datasets.moving_mnist import MovingMNIST
from src.datasets.moving_mnist_speed import MovingMNISTFixedSpeed
from src.evaluation.metrics import evaluate
from scripts.phase6_train import build_compact
from scripts.benchmark_efficiency import measure_latency


PARTS = {"encoder": "encoder.layers", "translator": "transition.layers",
         "decoder": "decoder.layers"}
BACKEND = "onednn"     # the only quantized engine in the Windows CPU build I used
# ConvTranspose2d has no per-channel weight quantization in this backend,
# so the direct check below uses one scale for the whole weight tensor
PER_TENSOR = QConfig(activation=HistogramObserver.with_args(reduce_range=False),
                     weight=default_weight_observer)


def check_convtranspose_kernel():
    """One int8 ConvTranspose2d layer against fp32 (wrong outputs on my build)."""
    from torch.ao.nn.quantized import ConvTranspose2d as QConvTranspose2d
    torch.manual_seed(0)
    ct = nn.ConvTranspose2d(16, 8, 4, 2, 1).eval()
    x = torch.randn(1, 16, 8, 8)
    ref = ct(x)
    w_scale = ct.weight.abs().max().item() / 127
    out_scale = ref.abs().max().item() / 127
    q = QConvTranspose2d(16, 8, 4, 2, 1)
    q.set_weight_bias(torch.quantize_per_tensor(ct.weight.detach(), w_scale, 0, torch.qint8),
                      ct.bias.detach())
    q.scale, q.zero_point = out_scale * 2, 128
    x_scale = x.abs().max().item() / 127
    out = q(torch.quantize_per_tensor(x, x_scale, 128, torch.quint8)).dequantize()
    return float(((out - ref) ** 2).mean() / (ref ** 2).mean())


def get_sub(model, path):
    obj = model
    for name in path.split("."):
        obj = getattr(obj, name)
    return obj


def set_sub(model, path, value):
    parent, name = path.rsplit(".", 1)
    setattr(get_sub(model, parent), name, value)


def quantize(model, parts, calib, layer=None):
    """
    Quantize the given parts (list of keys of PARTS). If layer is given as
    (part, index), only that single layer of that part is quantized.
    """
    qmodel = copy.deepcopy(model).eval()
    qconfig = get_default_qconfig_mapping(BACKEND).global_qconfig
    for part in parts:
        path = PARTS[part]
        seq = get_sub(qmodel, path)
        if layer is None:
            # transposed convs stay in fp32 (see check_convtranspose_kernel)
            mapping = get_default_qconfig_mapping(BACKEND).set_object_type(
                nn.ConvTranspose2d, None)
        else:
            mapping = QConfigMapping().set_global(None).set_module_name(str(layer[1]), qconfig)
        example = (torch.randn(1, *_input_shape(qmodel, part)),)
        set_sub(qmodel, path, prepare_fx(seq, mapping, example))
    with torch.no_grad():
        for inp in calib:
            qmodel(inp)
    for part in parts:
        path = PARTS[part]
        set_sub(qmodel, path, convert_fx(get_sub(qmodel, path)))
    return qmodel


def _input_shape(model, part):
    n = model.n_layers
    if part == "encoder":
        return (1, 64, 64)
    if part == "translator":
        return (640, 64 // 2 ** n, 64 // 2 ** n)
    return (64, 64 // 2 ** n, 64 // 2 ** n)


def weights_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 1e6


def quantized_modules(model):
    names = []
    for part, path in PARTS.items():
        for name, mod in get_sub(model, path).named_modules():
            if "quantized" in type(mod).__module__:
                names.append(f"{part}.{name}:{type(mod).__name__}")
    return names


def output_gap(model, ref_outputs, loader):
    """Relative MSE between this model's outputs and the fp32 outputs."""
    num, den = 0.0, 0.0
    with torch.no_grad():
        for (inp, _), ref in zip(loader, ref_outputs):
            out = model(inp)
            num += ((out - ref) ** 2).sum().item()
            den += (ref ** 2).sum().item()
    return num / den


def test_mse(model, loader):
    total, n = 0.0, 0
    with torch.no_grad():
        for inp, tgt in loader:
            total += ((model(inp).clamp(0, 1) - tgt) ** 2).mean().item() * inp.shape[0]
            n += inp.shape[0]
    return total / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--out", default="results/quantization_int8.json")
    args = parser.parse_args()

    torch.backends.quantized.engine = BACKEND
    convt_err = check_convtranspose_kernel()
    print(f"int8 ConvTranspose2d check, relative MSE vs fp32: {convt_err:.3f}")
    torch.manual_seed(0)

    model = build_compact("n3")
    ckpt = torch.load(Path(args.ckpt_root) / "compact-mmnist-otf-n3" / "best.pt",
                      map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    calib_ds = MovingMNISTFixedSpeed(Path(args.data_root) / "mnist", speed=None,
                                     n_samples=200, seed=500000, mnist_train=True)
    calib = [inp for inp, _ in DataLoader(calib_ds, batch_size=50)]
    test_ds = MovingMNIST(Path(args.data_root) / "moving_mnist" / "mnist_test_seq.npy",
                          split="test", input_len=10, target_len=10)
    test_loader = DataLoader(test_ds, batch_size=50, shuffle=False)
    with torch.no_grad():
        ref_outputs = [model(inp) for inp, _ in test_loader]

    configs = {"fp32": [], "encoder": ["encoder"], "translator": ["translator"],
               "decoder": ["decoder"], "all": list(PARTS)}
    results = {"torch_version": torch.__version__, "backend": BACKEND,
               "convtranspose_int8_check_rel_mse": convt_err}
    for name, parts in configs.items():
        print(f"\n== {name}", flush=True)
        qmodel = quantize(model, parts, calib) if parts else model
        res = evaluate(qmodel, test_loader, torch.device("cpu"))
        mse = [float(x) for x in res["mse_per_step"]]
        ssim = [float(x) for x in res["ssim_per_step"]]
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        lat = measure_latency(qmodel, (1, 10, 1, 64, 64))
        torch.set_num_threads(n_threads)
        results[name] = {
            "test_mse_avg": sum(mse) / 10, "test_ssim_avg": sum(ssim) / 10,
            "output_gap_rel_mse": output_gap(qmodel, ref_outputs, test_loader),
            "weights_mb": weights_size_mb(qmodel),
            "latency_median_ms": lat["median_ms"], "latency_p95_ms": lat["p95_ms"],
            "quantized_modules": quantized_modules(qmodel),
        }
        r = results[name]
        print(f"  MSE {r['test_mse_avg']:.5f}  SSIM {r['test_ssim_avg']:.4f}  "
              f"gap {r['output_gap_rel_mse']:.2e}  {r['weights_mb']:.2f} MB  "
              f"latency {r['latency_median_ms']:.2f} ms (p95 {r['latency_p95_ms']:.2f})")
        print(f"  quantized: {len(r['quantized_modules'])} modules")

    # Each conv layer on its own: which layer is the most sensitive?
    print("\n== one layer at a time", flush=True)
    per_layer = {}
    for part, path in PARTS.items():
        for idx, mod in enumerate(get_sub(model, path)):
            if not isinstance(mod, nn.Conv2d):     # ConvTranspose2d stays fp32
                continue
            qmodel = quantize(model, [part], calib, layer=(part, idx))
            key = f"{part}[{idx}] {type(mod).__name__} {tuple(mod.weight.shape)}"
            per_layer[key] = {
                "params": mod.weight.numel() + (mod.bias.numel() if mod.bias is not None else 0),
                "output_gap_rel_mse": output_gap(qmodel, ref_outputs, test_loader),
                "test_mse_avg": test_mse(qmodel, test_loader),
                "quantized_modules": quantized_modules(qmodel),
            }
            print(f"  {key:52s} gap {per_layer[key]['output_gap_rel_mse']:.2e}  "
                  f"MSE {per_layer[key]['test_mse_avg']:.5f}", flush=True)
    results["per_layer"] = per_layer

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
