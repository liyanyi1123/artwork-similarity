# Synthetic example

`synthetic.png` is a 256 × 256 geometric image generated during packaging from a plain background, rectangle, ellipse, triangle and diagonal line. It contains no dataset artwork. It supports a small transform/extraction smoke test and replaces the original static prototype's demo artwork.

```bash
python scripts/generate_transforms.py examples/synthetic.png outputs/transforms
python scripts/extract_vectors.py examples outputs/vectors
```

The example is not included in any reported research metric.
