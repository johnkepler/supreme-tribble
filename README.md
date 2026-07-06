# rfdetr-to-coreml

Convert [RF-DETR](https://github.com/roboflow/rf-detr) PyTorch weights
(`weights.pt`) into a native CoreML `.mlpackage` that runs on the Apple Neural
Engine / GPU.

RF-DETR ships an ONNX exporter but no CoreML one (Roboflow's hosted CoreML export
is a paid feature). Running the ONNX model on-device through ONNX-Runtime's CoreML
execution provider strands the deformable-attention decoder on the CPU (~1–2 fps).
This tool produces a **native** CoreML mlprogram instead, so Core ML can place the
backbone and decoder on the ANE/GPU — a legitimate real-time path (~15 fps on an A17).

The hard part is the deformable-attention decoder, which `coremltools` can't trace
as-is. The converter applies four trace-time patches (a rank-safe attention
rewrite using `grid_sample`, safe `int`/`bool` casts, antialias-off interpolation,
and a flattened `meshgrid`) and **numerically validates** the rewritten attention
against the stock forward (max|Δ| < 1e-3) before converting — so the export is
faithful.

## Install

numpy **must** stay below 2.4 (2.4 breaks coremltools' CoreML export). Tested on
Python 3.13, torch 2.7, coremltools 9.0.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you need a specific RF-DETR commit, install it editable from a local checkout
instead of the PyPI pin: `pip install -e /path/to/rf-detr`.

## Usage

For a standard RF-DETR **Small** checkpoint, just pass the weights — variant
defaults to `small`, resolution to the variant's native size, and `num_classes`
is inferred from the checkpoint's class head:

```bash
python rfdetr_to_coreml.py path/to/weights.pt
# -> writes path/to/weights.mlpackage
```

Full control:

```bash
python rfdetr_to_coreml.py weights.pt \
    --output models/detector.mlpackage \
    --variant small \
    --resolution 640 \ # MUST match the resolution the model was trained at
    --num-classes 7 \
    --class-names "person,bicycle,car,motorcycle,bus,truck,dog" \
    --deployment-target iOS16 \
    --image-name image --boxes-name boxes --logits-name logits
```

Run `python rfdetr_to_coreml.py --help` for the full flag list.

### Class labels

If the checkpoint records class names (the standard Roboflow training flow stores
them in `args["class_names"]`), they're read automatically and embedded in the
`.mlpackage` metadata. If your checkpoint doesn't have them, pass them explicitly
in logit order (background excluded):

```bash
python rfdetr_to_coreml.py weights.pt \
    --class-names "person,bicycle,car"
```

The count is cross-checked against `num_classes`; on a mismatch the tool warns and
skips embedding rather than ship wrong labels. Embedded names land under the
metadata key `classes` (comma-joined), readable on-device via
`MLModel.modelDescription.metadata[.creatorDefinedKey]`.

**Sparse label spaces:** empty entries are kept as positional gaps, so a layout
like COCO's — 90 logit slots with 10 unused ids — maps straight through:

```bash
python rfdetr_to_coreml.py weights.pt \
    --num-classes 90 \
    --class-names "person,bicycle,car,...,,,...,toothbrush"  # gaps preserved
```

The name for logit index *i* sits at entry *i-1* (index 0 is background, no
entry). The count must still equal `num_classes` (90 here), so pass the gaps.

### Important: resolution must match training

`--resolution` defaults to the variant's *native* config resolution (Nano 384,
Small 512, Medium 576, Large 704). If you trained at a different resolution
(e.g. Small fine-tuned at 640), you **must** pass the value you trained with
— otherwise the position embeddings mismatch and detections degrade.
The checkpoint does not record this, so the tool can't infer it.

## Output contract

| Feature  | Kind   | Shape                        | Meaning |
|----------|--------|------------------------------|---------|
| `image`  | image  | `1×3×R×R` RGB                 | Feed raw `[0,255]` pixels. The graph bakes in `scale=1/255` + ImageNet normalization. |
| `boxes`  | tensor | `1×num_queries×4`            | `cxcywh`, normalized `[0,1]`. |
| `logits` | tensor | `1×num_queries×(num_classes+1)` | Pre-sigmoid. Index 0 is the background/placeholder slot. |

This is the same I/O as RF-DETR's ONNX export, so existing post-processing
(sigmoid → per-query top class → threshold → cxcywh→xyxy) decodes either format.
Output feature names are configurable via the `--*-name` flags.

## Palettization (smaller models)

`--palettize-bits {4,6,8}` k-means-clusters the weights into a lookup table for a
smaller `.mlpackage` — 6-bit is roughly **2.6× smaller** (e.g. ~52 MB → ~20 MB on
Nano) and, on the checkpoints we've measured, detection-for-detection equivalent.

Palettization is **lossy**, so the tool won't take "it loaded" for validation.
Pass real images and it runs an accuracy gate — predicts with the fp16 and the
palettized model on each, and **fails the export** unless detections match
(identical count above threshold, top confidence within ±0.02):

```bash
python rfdetr_to_coreml.py weights.pt \
    --palettize-bits 6 \
    --eval-images samples/*.jpg \
    --eval-conf-threshold 0.5
```

Without `--eval-images` it still palettizes but prints a loud warning that the
result was **not** validated. Requires `scikit-learn` (see [Install](#install)).

**Deployment target:** below iOS18 the palettized (LUT) weights may be expanded
back to float at load — outputs stay correct, but you keep the on-disk size win
without the runtime memory/bandwidth benefit. Raise `--deployment-target` if you
need the weights to stay compressed at runtime.

## Supported variants

`nano`, `small`, `medium`, `base`, `large` via `--variant`.

**Limitation:** the rank-safe attention rewrite is specialized for single-scale
decoders (`n_levels == 1`), which covers **Nano and Small**. Multi-scale variants
raise a clear error rather than emitting a silently-wrong model. Extending to
multi-scale means generalizing `_single_scale_deform_attn_forward` to loop over levels.

**Note:** If you want to convert all variants incl. Medium/Base/Large, check out this recent repo: landchenxuan/rf-detr-to-coreml

## Using as a library

`convert(...)` returns the CoreML `MLModel`. Pass `save=False` to get the
in-memory model without writing it, then post-process (e.g. palettize, add
metadata) and save once — no save/reload/re-save round-trip:

```python
import coremltools as ct
import coremltools.optimize.coreml as cto
from rfdetr_to_coreml import convert, infer_num_classes

model = convert(
    weights="weights.pt", out=None, variant="nano", resolution=384,
    num_classes=infer_num_classes("weights.pt") or 90,
    deploy_target=ct.target.iOS16,
    image_name="image", boxes_name="boxes", logits_name="logits",
    verify=False, save=False,          # <- return the MLModel, don't write it
)
model = cto.palettize_weights(
    model, cto.OptimizationConfig(cto.OpPalettizerConfig(mode="kmeans", nbits=6)))
model.save("detector.mlpackage")
```

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The single-scale deformable-attention specialization is derived from RF-DETR's
own Apache-2.0 implementation (`MSDeformAttn.forward` /
`ms_deform_attn_core_pytorch`, itself from LW-DETR and Deformable DETR); see
NOTICE for full attribution. The CoreML conversion approach was independently
implemented against the public coremltools APIs. The
[iacomus/roboflow-weedcrop-edge-demo](https://github.com/iacomus/roboflow-weedcrop-edge-demo)
project demonstrates the same approach and is acknowledged as prior art.
