---
title: GRDFNet
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 5.44.1
app_file: app.py
license: mit
short_description: A lightweight image restoration architecture
pinned: true
---

# GRDFNet

GRDFNet is a lightweight image restoration network that combines gated and dilated residual blocks to deliver strong perceptual quality with modest compute requirements.

## Recommended Configurations

- `num_sets = 3`, `feature_channels = 32`: strong quality while staying fast for most desktop workloads.
- `num_sets = 6`, `feature_channels = 48`: highest quality configuration; expect roughly a 4x slowdown versus the 32-channel model.
- `num_sets = 3`, `feature_channels = 24`: suggested for lightly compressed video inference; typically 50~75% faster than the 32-channel variant when deployed with TensorRT.

## Performance Snapshot

Example TensorRT run on an NVIDIA RTX 4080 Super (16 GB):
```
DEBUG: TensorRT initialized. Setting shape.
DEBUG: Shape set. Getting output shape.
[INFO] Input: 1280x720 -> 1280x720 -> ModelOut: 1280x720 @ 30000/1001 fps
DEBUG: Before NVENC initialization.
[prof] frames=209 avg=208.2 fps
[prof] frames=431 avg=215.1 fps
[prof] frames=654 avg=217.3 fps
[INFO] Processed 709 frames in 3.280s -> 216.2 FPS
```

## Resources

- Model weights: https://huggingface.co/nicholasLane/GRDFNet
- Hosted demo: https://huggingface.co/spaces/nicholasLane/GRDFNet
