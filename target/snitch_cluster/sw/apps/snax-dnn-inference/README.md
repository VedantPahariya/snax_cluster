# SNAX DNN Inference Kernel

Multi-layer neural network inference on SNAX cluster with GEMMX hardware acceleration.

## Quick Start

```bash
cd target/snitch_cluster/

# Build RTL + Software
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson rtl-gen
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw

# Run inference
./bin/snitch_cluster.vlt ./sw/apps/snax-dnn-inference/build/snax-dnn-inference.elf
```

## Model Architecture

2-layer neural network:
- **Conv1** (7×7, stride=2) → Sigmoid → MatMul (GEMMX)
- **Conv2** (3×3, stride=1) → Sigmoid → MatMul (GEMMX)
- **I/O**: 1×1×128×128 → 1×16×64×64

## Files

- **include/dnn_ops.h** - Operation declarations
- **include/dnn_model.h** - Model structures
- **include/dnn_layers.c** - Operation implementations
- **src/main.c** - Inference kernel
- **src/dnn_model.c** - Model setup
- **src/params.hjson** - Configuration (modify to change model)
- **datagen.py** - Test data generator

## Customization

Edit `src/params.hjson` to:
- Add/remove layers
- Change layer dimensions
- Modify filter sizes and strides
- Adjust precision (FP64, FP32, FP16, FP8)

Then regenerate test data:
```bash
./datagen.py -c src/params.hjson -o src/data.h
```

## Documentation

See **DNN_INFERENCE_HANDBOOK.md** for:
- Complete architecture details
- API reference
- Build/run instructions
- Performance analysis
- Future extensions
- Troubleshooting guide

## Key Features

✅ Multi-layer inference
✅ GEMMX hardware acceleration support
✅ Modular operation library
✅ Automatic test data generation
✅ Performance monitoring
✅ Cluster synchronization
✅ Result verification

## Configuration

Currently optimized for: **snax_KUL_cluster.hjson**
- GEMMX accelerator available
- 56 TCDM ports
- 8×8 matrix multiply engine

## Support

Refer to inline documentation in:
- `dnn_ops.h` - Function specifications
- `dnn_model.h` - Model API
- `main.c` - Execution flow
- Source code comments

---

**For detailed information, see the comprehensive handbook: `DNN_INFERENCE_HANDBOOK.md`**
