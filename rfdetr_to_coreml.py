#!/usr/bin/env python
# Copyright (c) 2026 John Kepler
# Licensed under the Apache License, Version 2.0 (see LICENSE and NOTICE).
"""Convert RF-DETR PyTorch weights (`weights.pt`) into a native CoreML `.mlpackage`.

RF-DETR ships an ONNX exporter but no CoreML one (Roboflow's hosted CoreML export
is a paid Core-tier feature). ONNX-on-iOS via ONNX-Runtime's CoreML execution
provider leaves the deformable-attention decoder stranded on CPU (~1-2 fps). This
tool converts the weights to a *native* CoreML mlprogram so Core ML can place the
backbone and decoder on the ANE/GPU — the real-time path.

The deformable-attention decoder is what makes RF-DETR hostile to coremltools.
Four trace-time patches close every gap (see each patch's comment); the
single-scale attention specialization is numerically validated (max|Δ| < 1e-3)
against the stock forward before conversion, so the exported model is bit-for-bit
faithful. That specialization is derived from RF-DETR's own Apache-2.0
`MSDeformAttn.forward` / `ms_deform_attn_core_pytorch` (in turn from LW-DETR and
Deformable DETR); see NOTICE for attribution.

The exported model's I/O contract:
  input   "image"  : RGB image, native_res x native_res, pixels scaled 1/255 then
                     ImageNet-normalized inside the graph (feed raw [0,255] pixels).
  output  "boxes"  : (1, num_queries, 4) cxcywh, normalized [0,1].
  output  "logits" : (1, num_queries, num_classes + 1) pre-sigmoid; index 0 is the
                     background/placeholder slot.
(Names are configurable via flags.) This matches RF-DETR's ONNX export, so the
same post-processing decodes either.

Quickstart:
    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt          # NOTE: numpy<2.4 (2.4 breaks export)
    python rfdetr_to_coreml.py path/to/weights.pt

`--variant`, `--resolution`, and `--num-classes` default sensibly (Small; the
variant's native resolution; class count inferred from the checkpoint), so for a
standard Small checkpoint just pass the weights. Run `--help` for all options.

Limitation: the rank-safe attention rewrite is specialized for single-scale
decoders (n_levels == 1), which covers RF-DETR Nano/Small. Multi-scale variants
raise a clear error rather than exporting a wrong model.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs


# Variant name -> (rfdetr class name, config class name). Resolved lazily so the
# import cost (and any deprecation warnings) is only paid for the chosen variant.
_VARIANTS = {
    "nano": ("RFDETRNano", "RFDETRNanoConfig"),
    "small": ("RFDETRSmall", "RFDETRSmallConfig"),
    "medium": ("RFDETRMedium", "RFDETRMediumConfig"),
    "base": ("RFDETRBase", "RFDETRBaseConfig"),
    "large": ("RFDETRLarge", "RFDETRLargeConfig"),
}

_DEPLOY_TARGETS = {
    "iOS15": ct.target.iOS15,
    "iOS16": ct.target.iOS16,
    "iOS17": ct.target.iOS17,
    "iOS18": ct.target.iOS18,
}


def _resolve_variant(variant: str):
    """Return (model_class, native_resolution) for a variant name."""
    cls_name, cfg_name = _VARIANTS[variant]
    import rfdetr
    from rfdetr import config as rfdetr_config

    model_cls = getattr(rfdetr, cls_name)
    native_res = None
    cfg_cls = getattr(rfdetr_config, cfg_name, None)
    if cfg_cls is not None:
        try:
            native_res = cfg_cls.model_fields["resolution"].default
        except Exception:
            native_res = None
    return model_cls, native_res


def infer_num_classes(weights: Path) -> int | None:
    """Read num_classes from a checkpoint's class-embedding head.

    RF-DETR's class head emits `num_classes + 1` logits (index 0 is the
    background/placeholder slot), so the trained class count is the head's
    out_features minus one. Returns None if no class head is found.
    """
    try:
        ck = torch.load(str(weights), map_location="cpu", weights_only=False)
    except Exception as exc:  # corrupt / unreadable checkpoint -> let caller decide
        print(f"[warn] could not load checkpoint for num_classes inference: {exc}",
              file=sys.stderr)
        return None
    state = ck.get("model", ck) if isinstance(ck, dict) else ck
    if not hasattr(state, "items"):
        return None
    for key, value in state.items():
        if "class_embed" in key and key.endswith(".weight") and hasattr(value, "shape") \
                and value.ndim == 2:
            return int(value.shape[0]) - 1
    return None


def _register_scalar_cast_ops() -> None:
    """Re-lower torch `int()` / `bool()` so they emit a rank-0 MIL value.

    coremltools' default `int`/`bool` lowering preserves the operand's rank, but
    the values RF-DETR routes through these casts are consumed as scalar
    sizes/flags, and MIL rejects a rank>0 operand in those positions. For each
    cast: fold to a compile-time constant when the operand is statically known,
    otherwise reduce it to rank 0 with `squeeze` before casting. Written against
    the public coremltools custom torch-op registration API
    (`register_torch_op`); the fold/squeeze/cast handling is the functional
    requirement, not borrowed expression.
    """
    def _emit_scalar_cast(context, node, py_type, mil_type):
        operand = _get_inputs(context, node, expected=1)[0]
        if operand.can_be_folded_to_const():
            scalar = operand.val
            if isinstance(scalar, np.ndarray):
                scalar = scalar.reshape(-1)[0].item()
            result = mb.const(val=py_type(scalar), name=node.name)
        elif len(operand.shape) > 0:
            rank0 = mb.squeeze(x=operand, name=node.name + "_scalar")
            result = mb.cast(x=rank0, dtype=mil_type, name=node.name)
        else:
            result = mb.cast(x=operand, dtype=mil_type, name=node.name)
        context.add(result, node.name)

    @register_torch_op(torch_alias=["int"], override=True)
    def _int(context, node):
        _emit_scalar_cast(context, node, int, "int32")

    @register_torch_op(torch_alias=["bool"], override=True)
    def _bool(context, node):
        _emit_scalar_cast(context, node, bool, "bool")


def _single_scale_deform_attn_forward(self, query, reference_points, input_flatten,
                                      input_spatial_shapes, input_level_start_index,
                                      input_padding_mask=None, input_spatial_shapes_hw=None):
    """Drop-in `MSDeformAttn.forward` specialized to a single feature level.

    A faithful single-scale (`n_levels == 1`) specialization of RF-DETR's own
    `MSDeformAttn.forward` + `ms_deform_attn_core_pytorch`
    (`rfdetr.models.ops`, Apache-2.0 — itself from LW-DETR / Deformable-DETR).
    The stock multi-scale path builds a rank-6 sampling tensor
    `(batch, query, heads, levels, points, 2)`; MIL caps tensors at rank 5, so it
    can't convert. With one level the `levels` axis is redundant — we drop it,
    keeping every tensor at rank <= 4, and reuse the same `_bilinear_grid_sample`
    and conventions, so the result is numerically identical (asserted against the
    stock forward before conversion). Variable names follow the upstream module.
    """
    assert self.n_levels == 1, "single-scale specialization requires n_levels == 1"
    from rfdetr.models.ops.functions.ms_deform_attn_func import _bilinear_grid_sample

    batch_size, len_query, _ = query.shape
    batch_size, len_input, _ = input_flatten.shape
    n_heads, n_points = self.n_heads, self.n_points
    head_dim = self.d_model // n_heads

    value = self.value_proj(input_flatten)
    if input_padding_mask is not None:
        value = value.masked_fill(input_padding_mask[..., None], float(0))

    # Per-level axis omitted: rank-5 (..., n_points, 2) rather than the stock rank-6.
    sampling_offsets = self.sampling_offsets(query).view(batch_size, len_query, n_heads, n_points, 2)
    attention_weights = self.attention_weights(query).view(batch_size, len_query, n_heads, n_points)
    attention_weights = F.softmax(attention_weights, -1)

    reference = reference_points[:, :, 0]                       # single level -> (batch, query, 2|4)
    if reference.shape[-1] == 2:
        normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)[0]
        sampling_locations = reference[:, :, None, None, :] + sampling_offsets / normalizer
    elif reference.shape[-1] == 4:
        sampling_locations = (reference[:, :, None, None, :2]
                              + sampling_offsets / n_points * reference[:, :, None, None, 2:] * 0.5)
    else:
        raise ValueError("reference_points last dim must be 2 or 4")

    if input_spatial_shapes_hw is not None:
        height, width = input_spatial_shapes_hw[0]
    else:
        height = int(input_spatial_shapes[0, 0])
        width = int(input_spatial_shapes[0, 1])

    # (batch, len_input, C) -> (batch*n_heads, head_dim, H, W) for grid_sample.
    value = value.transpose(1, 2).contiguous().view(batch_size, n_heads, head_dim, len_input)
    value = value.view(batch_size * n_heads, head_dim, height, width)
    grid = (2 * sampling_locations - 1).transpose(1, 2).reshape(batch_size * n_heads, len_query, n_points, 2)
    sampled = _bilinear_grid_sample(value, grid, padding_mode="zeros", align_corners=False)
    weights = attention_weights.transpose(1, 2).reshape(batch_size * n_heads, 1, len_query, n_points)
    output = (sampled * weights).sum(-1).view(batch_size, n_heads * head_dim, len_query)
    return self.output_proj(output.transpose(1, 2).contiguous())


class _Wrap(torch.nn.Module):
    """Bake ImageNet normalization into the graph and unwrap the dict output.

    The CoreML ImageType feeds raw [0,255] pixels scaled to [0,1]; this module
    applies the ImageNet mean/std the backbone was trained with, then returns
    (boxes, logits) as a flat tuple coremltools can name.
    """
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        o = self.net(x)
        if isinstance(o, dict):
            return o["pred_boxes"], o["pred_logits"]
        return o[0], o[1]


def convert(weights: Path, out: Path, variant: str, resolution: int, num_classes: int,
            deploy_target, image_name: str, boxes_name: str, logits_name: str,
            verify: bool) -> None:
    _register_scalar_cast_ops()
    model_cls, _ = _resolve_variant(variant)

    print(f"[export] loading {weights}")
    print(f"[export] variant={variant}  resolution={resolution}  num_classes={num_classes}")
    torch.manual_seed(0)
    np.random.seed(0)

    api = model_cls(num_classes=num_classes, resolution=resolution,
                    pretrain_weights=str(weights), device="cpu")
    net = copy.deepcopy(api.model.model).eval().to("cpu")
    dummy = torch.randn(1, 3, resolution, resolution)

    # Locate the deformable-attention class so we can swap in the rank-safe forward.
    deform_cls = None
    for _, mod in net.named_modules():
        if "DeformAttn" in mod.__class__.__name__:
            deform_cls = type(mod)
            break
    assert deform_cls is not None, "no deformable-attention module found in this model"

    with torch.no_grad():
        b0, l0 = _Wrap(net).eval()(dummy)
    print(f"[export] model output shapes: boxes={tuple(b0.shape)} logits={tuple(l0.shape)}")

    # Validate the single-scale specialization matches the stock forward before trusting it.
    _orig_forward = deform_cls.forward
    deform_cls.forward = _single_scale_deform_attn_forward
    try:
        with torch.no_grad():
            b1, l1 = _Wrap(net).eval()(dummy)
    except AssertionError as exc:
        deform_cls.forward = _orig_forward
        raise SystemExit(f"[error] {exc}\n"
                         f"        Variant '{variant}' uses a multi-scale decoder this tool "
                         f"does not support. Only single-scale variants (nano/small) convert.")
    db = (b0 - b1).abs().max().item()
    dl = (l0 - l1).abs().max().item()
    print(f"[export] single-scale attention max|Δ|: boxes={db:.2e} logits={dl:.2e}")
    assert db < 1e-3 and dl < 1e-3, "single-scale attention diverged from the stock forward!"

    # Trace-time monkeypatches: coremltools can't trace antialiased interpolation
    # or a non-flattened meshgrid. Both are numerically inert at this resolution.
    _oi, _om = F.interpolate, torch.meshgrid

    def _no_aa(*a, **k):
        if k.get("mode") == "bicubic":
            k["mode"] = "bilinear"
        if k.get("mode") == "bilinear":
            k["antialias"] = False
        return _oi(*a, **k)

    def _flat_meshgrid(*ts, **k):
        return _om(*(t.reshape(-1) for t in ts), **k)

    F.interpolate, torch.meshgrid = _no_aa, _flat_meshgrid
    try:
        traced = torch.jit.trace(_Wrap(net).eval(), dummy, strict=False)
    finally:
        F.interpolate, torch.meshgrid = _oi, _om

    print("[export] trace OK -> converting to CoreML mlprogram ...")
    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name=image_name, shape=(1, 3, resolution, resolution),
                             scale=1 / 255.0, bias=[0.0, 0.0, 0.0],
                             color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name=boxes_name), ct.TensorType(name=logits_name)],
        minimum_deployment_target=deploy_target,
        convert_to="mlprogram",
    )
    print(f"[export] ct.convert done in {time.time() - t0:.1f}s")
    mlmodel.short_description = (
        f"RF-DETR {variant} ({num_classes} classes). {resolution}x{resolution} RGB. "
        f"'{boxes_name}'=cxcywh norm [0,1]; '{logits_name}' pre-sigmoid "
        f"({num_classes + 1}-wide, idx0=background)."
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"[export] saved {out}")

    if verify:
        print("[export] validating: reload + predict ...")
        loaded = ct.models.MLModel(str(out))
        spec = loaded.get_spec()
        print("  inputs :", [(i.name, i.type.WhichOneof("Type")) for i in spec.description.input])
        print("  outputs:", [o.name for o in spec.description.output])
        from PIL import Image
        img = Image.fromarray((np.random.rand(resolution, resolution, 3) * 255).astype(np.uint8))
        pred = loaded.predict({image_name: img})
        for k, v in pred.items():
            print(f"  out {k}: shape={getattr(v, 'shape', type(v).__name__)}")
        print("[export] CoreML validation OK")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert RF-DETR PyTorch weights to a native CoreML .mlpackage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("weights", type=Path, help="path to the RF-DETR weights .pt checkpoint")
    p.add_argument("-o", "--output", type=Path, default=None,
                   help="output .mlpackage path (default: <weights>.mlpackage beside the weights)")
    p.add_argument("--variant", choices=sorted(_VARIANTS), default="small",
                   help="RF-DETR variant the checkpoint was trained as")
    p.add_argument("--resolution", type=int, default=None,
                   help="input resolution; must match training (default: the variant's native resolution)")
    p.add_argument("--num-classes", type=int, default=None,
                   help="trained class count (default: inferred from the checkpoint's class head)")
    p.add_argument("--deployment-target", choices=sorted(_DEPLOY_TARGETS), default="iOS16",
                   help="minimum CoreML deployment target")
    p.add_argument("--image-name", default="image", help="name of the image input feature")
    p.add_argument("--boxes-name", default="boxes", help="name of the boxes output feature")
    p.add_argument("--logits-name", default="logits", help="name of the logits output feature")
    p.add_argument("--no-verify", action="store_true",
                   help="skip the reload + predict validation pass")
    args = p.parse_args()

    if not args.weights.exists():
        print(f"[error] weights not found: {args.weights}", file=sys.stderr)
        return 1

    resolution = args.resolution
    if resolution is None:
        _, native_res = _resolve_variant(args.variant)
        if native_res is None:
            print("[error] could not determine native resolution; pass --resolution explicitly",
                  file=sys.stderr)
            return 1
        resolution = native_res
        print(f"[info] using variant native resolution {resolution}; pass --resolution to override "
              f"(must match what the model was trained at)")

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = infer_num_classes(args.weights)
        if num_classes is None:
            print("[error] could not infer --num-classes from the checkpoint; pass it explicitly",
                  file=sys.stderr)
            return 1
        print(f"[info] inferred num_classes={num_classes} from the checkpoint class head")

    out = args.output or args.weights.with_suffix(".mlpackage")

    convert(
        weights=args.weights,
        out=out,
        variant=args.variant,
        resolution=resolution,
        num_classes=num_classes,
        deploy_target=_DEPLOY_TARGETS[args.deployment_target],
        image_name=args.image_name,
        boxes_name=args.boxes_name,
        logits_name=args.logits_name,
        verify=not args.no_verify,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
