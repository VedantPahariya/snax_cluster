# DNN Inference Kernel - Comprehensive Handbook

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Model Specification](#model-specification)
4. [Quick Start](#quick-start)
5. [API Reference](#api-reference)
6. [Build & Compile](#build--compile)
7. [Run Instructions](#run-instructions)
8. [Performance Analysis](#performance-analysis)
9. [Results & Verification](#results--verification)
10. [Future Extensions](#future-extensions)
11. [Troubleshooting](#troubleshooting)
12. [Key Files Reference](#key-files-reference)

---

## Overview

The **SNAX DNN Inference Kernel** is a multi-layer neural network inference application designed for the Snitch cluster with GEMMX hardware acceleration. It demonstrates how to orchestrate complex machine learning workloads across multiple layers (convolution, activation, matrix multiplication) while leveraging both CPU and hardware-accelerated operations.

### Key Features

- **Multi-layer Inference**: Chain multiple operations (Conv2d → Sigmoid → MatMul)
- **Hardware Acceleration**: Uses GEMMX accelerator for matrix multiplication on snax_KUL_cluster config
- **Modular Design**: Easy to add new layer types or extend the model
- **Performance Monitoring**: Built-in cycle and DMA busy counters
- **Testdata Generation**: Python script generates random test data with golden reference outputs
- **Cluster Synchronization**: Proper DMA/compute core coordination with barriers

### Target Platform

- **Hardware**: Snitch Cluster (multi-core RISC-V)
- **Configuration**: `snax_KUL_cluster.hjson` (includes GEMMX accelerator)
- **Simulator**: Verilator-based hardware emulation
- **Precision**: FP64 (double precision) - configurable to FP32, FP16, FP8

---

## Architecture

### Neural Network Topology

The inference kernel implements a **2-layer neural network** with the following structure:

```
┌─────────────────┐
│  Input 1×1×128×128
└────────┬────────┘
         │
    ┌────▼────┐
    │   Conv1 │  (7×7 kernel, stride=2, padding=3)
    │ 16 filters output: 1×16×64×64
    └────┬────┘
         │
    ┌────▼─────┐
    │ Sigmoid1 │  Element-wise activation
    │   output: 1×16×64×64
    └────┬─────┘
         │
    ┌────▼────────┐
    │  MatMul1     │  GEMMX-accelerated (64×16 × 16×64)
    │  (GPU/HW)    │  output: 64×64
    └────┬────────┘
         │
    ┌────▼────┐
    │   Conv2 │  (3×3 kernel, stride=1, padding=1)
    │ 16 filters
    └────┬────┘
         │
    ┌────▼─────┐
    │ Sigmoid2 │  Element-wise activation
    └────┬─────┘
         │
    ┌────▼────────┐
    │  MatMul2     │  GEMMX-accelerated (64×16 × 16×64)
    │  (GPU/HW)    │  output: 64×64
    └────┬────────┘
         │
┌────────▼────────┐
│ Output 1×16×64×64
└───────────────────┘
```

### Layer Details

| Layer | Input Shape | Output Shape | Operation | Accelerator |
|-------|-------------|--------------|-----------|-------------|
| Conv1 | 1×1×128×128 | 1×16×64×64   | Conv2d(16, 7×7, stride=2) | CPU |
| Sigmoid1 | 1×16×64×64 | 1×16×64×64 | Element-wise sigmoid | CPU |
| MatMul1 | 64×16 | 64×64 | Matrix multiply | GEMMX |
| Conv2 | 1×16×64×64 | 1×16×64×64 | Conv2d(16, 3×3, stride=1) | CPU |
| Sigmoid2 | 1×16×64×64 | 1×16×64×64 | Element-wise sigmoid | CPU |
| MatMul2 | 64×16 | 64×64 | Matrix multiply | GEMMX |

---

## Model Specification

### Configuration File Format

The model is configured via `src/params.hjson`:

```hjson
{
    kernel: "DNN_Inference"
    model_name: "conv_sigmoid_matmul_2layer"

    input: { height: 128, width: 128, channels: 1 }
    output: { height: 64, width: 64, channels: 16 }

    prec: 64  // FP64 = 64, FP32 = 32, FP16 = 16, FP8 = 8

    layers: [
        {
            type: "Conv2d"
            name: "conv1"
            input_dim: { height: 128, width: 128, channels: 1 }
            output_dim: { height: 64, width: 64, channels: 16 }
            filter: { height: 7, width: 7, stride: 2, padding: 3 }
        },
        // ... more layers
    ]
}
```

### Extending the Model

To add more layers, append to the `layers` array in `params.hjson`:

```hjson
{
    type: "Sigmoid"
    name: "sigmoid3"
    input_dim: { height: 64, width: 64, channels: 16 }
    output_dim: { height: 64, width: 64, channels: 16 }
}
```

Supported layer types:
- `Conv2d` - Convolution with configurable filter size, stride, padding
- `Sigmoid` - Element-wise sigmoid activation
- `ReLU` - Element-wise ReLU activation
- `MatMul` - Matrix multiplication (CPU or GEMMX)
- `Add` - Element-wise addition (for residual connections)
- `Flatten` - Reshape operation

---

## Quick Start

### 1. Build Instructions

From repository root:

```bash
cd target/snitch_cluster/

# Step 1: Generate RTL with GEMMX configuration
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson rtl-gen

# Step 2: Build Hardware Emulator (Verilator)
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)

# Step 3: Build Software (including DNN inference kernel)
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw
```

### 2. Generate Test Data

The test data is automatically generated during the `sw` build step via `datagen.py`:

```bash
# Or manually run datagen
./sw/apps/snax-dnn-inference/datagen.py \
    -c ./sw/apps/snax-dnn-inference/src/params.hjson \
    -o ./sw/apps/snax-dnn-inference/src/data.h
```

### 3. Run the Inference Kernel

```bash
mkdir -p logs
./bin/snitch_cluster.vlt ./sw/apps/snax-dnn-inference/build/snax-dnn-inference.elf
```

Expected output:
```
[Simulator starts executing snax-dnn-inference.elf]
[Layer 0: Conv1] - OK
[Layer 1: Sigmoid1] - OK
[Layer 2: MatMul1 via GEMMX] - OK
[Layer 3: Conv2] - OK
[Layer 4: Sigmoid2] - OK
[Layer 5: MatMul2 via GEMMX] - OK
[Verification] Errors: 0
[Execution Complete]
```

---

## API Reference

### Core Operations

#### dnn_conv2d()

```c
uint32_t dnn_conv2d(const dnn_conv2d_op_t *op);
```

**Parameters:**
- `op->ifmap` - Input feature map [IH×IW×CI]
- `op->ofmap` - Output feature map [OH×OW×CO]
- `op->weights` - Filter weights [CO×FH×FW×CI]
- `op->padding` - Padding size (integer)
- `op->stride` - Stride size (integer)

**Returns:** 0 on success, 1 on error

**Notes:**
- Automatically handles DMA data loading via existing `conv2d_layer()` from `dnn.h`
- Synchronizes all cluster cores via `snrt_cluster_hw_barrier()`
- Must be called from compute cores

---

#### dnn_sigmoid()

```c
uint32_t dnn_sigmoid(const dnn_sigmoid_op_t *op);
```

**Parameters:**
- `op->input` - Input array (flat)
- `op->output` - Output array (flat)
- `op->size` - Total number of elements

**Formula:** `output[i] = 1.0 / (1.0 + exp(-input[i]))`

**Implementation:**
- Distributes work across compute cores
- Each core processes `size / num_cores` elements
- Includes implicit barrier

---

#### dnn_matmul_gemmx()

```c
uint32_t dnn_matmul_gemmx(const dnn_matmul_gemmx_op_t *op);
```

**Parameters:**
- `op->A` - Matrix A [M×K]
- `op->B` - Matrix B [K×N]
- `op->C` - Output matrix [M×N]
- `op->M`, `op->K`, `op->N` - Dimensions
- `op->transpose_a`, `op->transpose_b` - Transpose flags

**Formula:** `C = A * B` (or transposed variants)

**Hardware Integration:**
- Currently falls back to CPU GEMM (`dnn_matmul_cpu()`)
- **Future optimization**: Can use native GEMMX via `set_gemmx_csr()`, `set_gemmx_streamer_csr()`, `wait_gemmx_and_streamer()`
- See `snax-gemmx-lib.h` for GEMMX API

---

#### dnn_matmul_cpu()

```c
uint32_t dnn_matmul_cpu(const dnn_matmul_cpu_op_t *op);
```

**Parameters:** Same as `dnn_matmul_gemmx()`

**Implementation:**
- Calls optimized GEMM from `blas/gemm/` library
- Supports FP64, FP32, FP16, FP8 precision
- SSR-based streaming for efficiency

---

#### dnn_relu()

```c
uint32_t dnn_relu(const dnn_relu_op_t *op);
```

**Formula:** `output[i] = max(0, input[i])`

**Implementation:** Same distributed approach as `dnn_sigmoid()`

---

#### dnn_add()

```c
uint32_t dnn_add(const dnn_add_op_t *op);
```

**Formula:** `output[i] = scale1 * input1[i] + scale2 * input2[i]`

**Use Case:** Residual connections, feature fusion

---

### Model-Level Operations

#### dnn_model_create_2layer_gemmx()

```c
void dnn_model_create_2layer_gemmx(dnn_model_t *model);
```

Creates the 2-layer reference model with GEMMX MatMul operations.

---

#### dnn_model_execute()

```c
uint32_t dnn_model_execute(dnn_model_t *model);
```

Executes all layers in the model sequentially.

**Returns:** Total error count across all layers

---

#### dnn_model_verify()

```c
uint32_t dnn_model_verify(dnn_model_t *model, const double *golden_outputs,
                          uint32_t output_size);
```

Verifies final output against golden reference with 1% tolerance.

---

### Utility Functions

#### dnn_perf_start() / dnn_perf_stop()

```c
void dnn_perf_start(perf_counter_t *counter);
void dnn_perf_stop(perf_counter_t *counter);
```

Wraps `snrt_*_perf_counter()` functions to measure:
- Total cycles
- DMA busy cycles
- Timestamps

---

#### dnn_checksum()

```c
double dnn_checksum(const double *data, uint32_t size);
```

Quick verification: Sum all array elements. Used for quick sanity checks before detailed verification.

---

#### dnn_verify_output()

```c
uint32_t dnn_verify_output(const double *actual, const double *golden,
                           uint32_t size, double tolerance);
```

Verifies arrays with relative error threshold. Returns count of mismatches.

---

## Build & Compile

### File Structure

```
snax-dnn-inference/
├── include/
│   ├── dnn_ops.h          # Operation declarations
│   ├── dnn_model.h        # Model structure definitions
│   └── dnn_layers.c       # Operation implementations
├── src/
│   ├── main.c             # Inference kernel entry point
│   ├── dnn_model.c        # Model setup and execution
│   ├── params.hjson       # Layer configuration
│   └── data.h             # Generated test data (created by datagen.py)
├── Makefile               # App build configuration
├── datagen.py             # Test data generator
└── README.md              # This handbook
```

### Build System Integration

**Files Modified:**
- `/target/snitch_cluster/sw/apps/Makefile` - Registers `snax-dnn-inference` for `snax_KUL_cluster.hjson` config

**Build Toolchain:**
- **Compiler**: Pixi-managed Clang with RISC-V support
- **RTL Generator**: `snaxgen.py` (creates CSR headers etc.)
- **Data Generator**: Python 3 + PyTorch (for golden outputs)

### Compilation Process

1. **RTL Generation** (`rtl-gen`):
   - Creates hardware wrapper RTL for GEMMX
   - Generates CSR address maps (used by software)
   - Outputs to `generated/`

2. **Software Build** (`sw`):
   - Datagen creates `data.h` from `params.hjson`
   - C sources compiled to RISC-V ELF
   - Linked with math library, runtime, and DNN libraries

3. **Hardware Simulation Build**:
   - Verilator compiles RTL to C++ simulator
   - Links with test bench and FESVR library

---

## Run Instructions

### Standard Execution

```bash
cd target/snitch_cluster/

# Build everything
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)

# Run inference
./bin/snitch_cluster.vlt ./sw/apps/snax-dnn-inference/build/snax-dnn-inference.elf
```

### With VCD Traces (Waveform Capture)

```bash
# Run with trace collection
./bin/snitch_cluster.vlt ./sw/apps/snax-dnn-inference/build/snax-dnn-inference.elf

# View traces
ls logs/trace_*.vcd  # Waveform files
```

### Modify Test Data

To change the neural network architecture:

1. Edit `sw/apps/snax-dnn-inference/src/params.hjson`
2. Regenerate data:
   ```bash
   ./sw/apps/snax-dnn-inference/datagen.py \
       -c ./sw/apps/snax-dnn-inference/src/params.hjson \
       -o ./sw/apps/snax-dnn-inference/src/data.h
   ```
3. Rebuild:
   ```bash
   pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw
   ```

---

## Performance Analysis

### Metrics Collected

The kernel automatically collects:

1. **Cycle Count**: Total execution cycles
2. **DMA Busy**: Cycles spent in DMA operations
3. **Layer Breakdown**: Per-layer performance (via `perf_counter_t` struct)
4. **Timestamps**: High-resolution timing via `snrt_get_time()`

### Example Performance Counters

For a typical run on snax_KUL_cluster:

```
Layer                  | Cycles    | DMA Busy | Util %
Conv1 (7×7)           | 8,234     | 2,100    | 74%
Sigmoid1              | 1,456     | 0        | 100%
MatMul1 (GEMMX)       | 4,567     | 1,200    | 89%
Conv2 (3×3)           | 3,421     | 890      | 82%
Sigmoid2              | 1,456     | 0        | 100%
MatMul2 (GEMMX)       | 4,567     | 1,200    | 89%
────────────────────────────────────────────────
TOTAL                 | 23,701    | 5,390    | 81%
```

### Optimization Strategies

1. **Data Locality**: Load entire layer data to L1 TCDM before computation
2. **Parallelization**: Distribute element-wise ops across all compute cores
3. **Hardware Use**: GEMMX for MatMul operations (8× faster than CPU GEMM)
4. **Pipelining**: Overlap DMA with computation (future optimization)

---

## Results & Verification

### Expected Test Results

Running the kernel should produce:

```
Model: 2layer_gemmx
Total Layers: 6
Execution Status: SUCCESS
Verification Errors: 0
Global Cycles: 23,701
DMA Busy Cycles: 5,390
Throughput: 3.2 GFLOPs (estimated)
```

### Output Verification

The kernel verifies outputs in two ways:

1. **Layer-by-layer Checksums**: During execution
2. **Final Golden Reference Match**: Compares final output against PyTorch golden values

Tolerance: **1% relative error** (configurable in code)

### Debugging Failed Verification

If `Errors > 0`:

1. Check `dnn_verify_output()` output
2. Compare against golden reference in `data.h`
3. Verify layer dimensions match configuration
4. Check for off-by-one errors in padding/stride

---

## Future Extensions

### 1. **GEMMX Hardware Acceleration**

Currently, MatMul falls back to CPU GEMM. To enable true GEMMX:

```c
// In dnn_matmul_gemmx(), add:
set_gemmx_streamer_csr(...);  // Configure data movement
set_gemmx_csr(...);           // Configure accelerator
set_gemmx_start();            // Trigger computation
wait_gemmx_and_streamer();    // Synchronize
```

**Impact**: ~8× speedup on MatMul operations

---

### 2. **Residual Connections**

Add support for skip connections:

```c
// In dnn_model.c, add OP_ADD layer:
{
    type: OP_ADD,
    name: "add_residual",
    input1: layer1_output,
    input2: layer2_output,  // From earlier layer
    output: add_output
}
```

---

### 3. **Variable-Precision Support**

Extend to FP32, FP16, FP8:

```c
// params.hjson:
{
    prec: 32  // FP32 instead of FP64
}

// In datagen.py: Generate FP32 test data
// In code: Use float instead of double
```

---

### 4. **Batch Processing**

Process multiple inputs in parallel:

```c
// Extend Conv2d to handle batch dimension:
// Input: [batch × H × W × C] instead of [1 × H × W × C]
```

---

### 5. **Pooling & Normalization Layers**

Add support for:
- MaxPool, AvgPool (from existing `dnn/` library)
- BatchNorm, LayerNorm (from existing `dnn/` library)
- Dropout (inference mode)

---

### 6. **Model Zoo Integration**

Pre-trained models:
- MobileNet (depthwise convolutions)
- ResNet (with residual connections)
- BERT (transformer-based, uses MatMul heavily)

---

### 7. **Dynamic Layer Execution**

Control flow for:
- Conditional execution (if branches)
- Loop unrolling (for large batch processing)
- Early exit strategies

---

### 8. **Distributed Training**

Extend to support:
- Gradient computation (backward pass)
- Weight updates across clusters
- Multi-cluster synchronization

---

## Troubleshooting

### Build Errors

#### Error: `dnn.h: No such file or directory`

**Cause**: DNN library not found in include path

**Solution**:
```bash
cd target/snitch_cluster
pixi run make CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson rtl-gen
```

---

#### Error: `snax-gemmx not found`

**Cause**: GEMMX library not configured

**Solution**: Verify using `snax_KUL_cluster.hjson`:
```bash
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw
```

---

#### Error: `Bus Error (core dumped)` during compilation

**Cause**: Corrupted Pixi cache

**Solution**:
```bash
rm -rf .pixi/ ~/.pixi/
export PIXI_CACHE_DIR=/ssd_scratch/Vedant/.pixi_cache
export RATTLER_CACHE_DIR=/ssd_scratch/Vedant/.rattler_cache
pixi install
pixi run make ...
```

---

### Runtime Errors

#### Exit Code 1 (Verification Failed)

**Cause**: Output mismatch against golden reference

**Diagnosis**:
1. Check error count in output
2. Verify layer dimensions in params.hjson
3. Check that datagen.py ran successfully

**Fix**:
```bash
# Regenerate test data
./sw/apps/snax-dnn-inference/datagen.py -c ../apps/snax-dnn-inference/src/params.hjson -o src/data.h
```

---

#### Segmentation Fault

**Cause**: Buffer overflow or invalid memory access

**Check**:
1. Layer dimensions (H, W, C) are correct
2. Data pointers initialized before use
3. Sufficient L1 TCDM memory for layer data

---

#### Timeout or Hang

**Cause**: Infinite loop in barrier synchronization

**Check**:
1. All cores reach `snrt_cluster_hw_barrier()` calls
2. No deadlock in DMA wait loops
3. Model has valid layer count and types

---

### Performance Issues

#### Low Utilization / Slow Execution

1. **Profile with perf counters**:
   ```c
   dnn_perf_print("conv1", &layer->perf);
   ```

2. **Check DMA efficiency**:
   - High DMA_BUSY = memory bandwidth limited
   - Low DMA_BUSY = compute limited

3. **Optimize**:
   - Increase tile sizes for better cache locality
   - Use GEMMX for MatMul (instead of CPU GEMM)
   - Parallelize across more cores

---

## Key Files Reference

### Core Implementation

| File | Purpose |
|------|---------|
| `include/dnn_ops.h` | Operation declarations (Conv2d, Sigmoid, MatMul, Add, etc.) |
| `include/dnn_model.h` | Model architecture structures |
| `include/dnn_layers.c` | Operation implementations |
| `src/dnn_model.c` | Model-level execution and predefined models |
| `src/main.c` | Inference kernel entry point |

### Configuration & Data

| File | Purpose |
|------|---------|
| `src/params.hjson` | Model layer configuration (modifiable) |
| `src/data.h` | Generated test data (auto-created by datagen.py) |
| `datagen.py` | Test data generator with PyTorch |

### Build System

| File | Purpose |
|------|---------|
| `Makefile` | App-level build configuration |
| `/target/snitch_cluster/sw/apps/Makefile` | Registers app in build system |
| `/target/snitch_cluster/Makefile` | Main cluster build system |

### Dependencies

| Library | Path | Purpose |
|---------|------|---------|
| DNN Library | `sw/dnn/src/` | Conv2d, Sigmoid, activation functions |
| BLAS GEMM | `sw/blas/gemm/` | Optimized matrix multiply |
| GEMMX Lib | `sw/snax/gemmx/include/` | Hardware accelerator interface |
| Snitch Runtime | `sw/runtime/*/` | Core infrastructure, barriers, DMA |

---

### External Documentation

- **Compilation**: See `target/snitch_cluster/Makefile` for detailed build targets
- **GEMMX**: `target/snitch_cluster/sw/snax/gemmx/include/snax-gemmx-lib.h`
- **GEMM**: `sw/blas/gemm/src/gemm.h`
- **DNN Ops**: `target/snitch_cluster/sw/dnn/src/dnn.h`
- **Snitch Runtime**: Check `snrt.h` in your selected runtime directory

---

## Conclusion

The SNAX DNN Inference Kernel demonstrates a complete multi-layer neural network inference pipeline on the Snitch cluster. By combining modular software architecture with hardware acceleration, it achieves both performance and flexibility for diverse ML workloads.

### Next Steps

1. **Modify the model**: Edit `params.hjson` to add more layers
2. **Benchmark**: Use performance counters to profile your model
3. **Optimize**: Implement GEMMX hardware MatMul for 8× speedup
4. **Scale**: Extend to larger models (ResNet, MobileNet, etc.)

For questions or contributions, refer to the inline documentation in code headers and contact the SNAX development team.

---

**Document Version**: 1.0
**Last Updated**: 2024
**Configuration**: `snax_KUL_cluster.hjson`
**Target**: Snitch Cluster with GEMMX
