# SNAX Setup Guide: GEMMX Accelerator & Multiple ALU Cores

This guide covers setup of the OpenGEMM (GEMMX) accelerator and configurations for multiple ALU accelerators with single or multiple cores.

## Part 1: Setting Up GEMMX (OpenGEMM) Accelerator

### Quick Start - Build GEMMX Cluster

Run these commands from the root of your `snax_cluster` repository:

```bash
cd target/snitch_cluster/

# 1. Generate RTL with GEMMX configuration
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson rtl-gen -j$(nproc)

# 2. Build Hardware Emulator (Verilator)
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)

# 3. Build Software for GEMMX
pixi run make -C . CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson sw

# 4. Run the Emulator with GEMMX test
mkdir -p logs
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-matmul/build/snax-gemmx-matmul.elf
```

### GEMMX Configuration Explained

The file `/ssd_scratch/Vedant/snax_cluster/target/snitch_cluster/cfg/snax_KUL_cluster.hjson` provides:

**Key Parameters:**
- `snax_acc_name`: "snax_streamer_gemmX" - The GEMMX accelerator with streaming
- `bender_target`: "snax_gemmX" - Builds the GEMMX hardware module
- `snax_tcdm_ports`: 56 - Number of TCDM (Tightly Coupled Data Memory) ports
- `snax_gemmx_mesh_row`: 8 - GEMMX mesh dimensions (8x8)
- `snax_gemmx_mesh_col`: 8
- `snax_gemmx_tile_size`: 8 - Tile size for matrix operations
- `with_pipeline`: true - Enable pipelining for performance

**Streamer Configuration:**
- `spatial_bounds`: [[8], [8]] - 8x8 spatial tiles
- `temporal_dim`: [6, 3] - 6 and 3 temporal dimensions for reader and writer
- `has_transpose`: true - Hardware support for matrix transpose
- `has_C_broadcast`: true - Broadcast of C matrix for efficiency

### Available GEMMX Apps

Test with these pre-built applications:
- `snax-gemmx-matmul` - Basic matrix multiplication
- `snax-gemmx-conv` - Convolution operation

Run them with:
```bash
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-matmul/build/snax-gemmx-matmul.elf
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-conv/build/snax-gemmx-conv.elf
```

---

## Part 2: Multiple ALU Accelerators with One Core

### Overview

You CAN have **multiple instances of ALU accelerators** accessible from a **single compute core**. This is configured in the `snax_acc_cfg` array.

### Configuration Options

**Option 1: Multiple ALU instances (existing config)**

Use the provided configuration:
```bash
cd target/snitch_cluster/

# Build with multiple ALU accelerators
pixi run make -C . CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)

# Build software
pixi run make -C . CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson sw

# Run test
./bin/snitch_cluster.vlt ./sw/apps/snax-multi-alu/build/snax-multi-alu.elf
```

### How It Works

The `snax_acc_cfg` is an **array** of accelerators. Each entry allows you to:

1. **Add multiple instances** of the same accelerator type:
```hjson
snax_acc_cfg: [
    { snax_acc_name: "snax_alu", ... },  // ALU #0
    { snax_acc_name: "snax_alu", ... },  // ALU #1
    { snax_acc_name: "snax_alu", ... },  // ALU #2
]
```

2. Each ALU instance has its own:
   - CSR (Control Status Register) addresses
   - TCDM port connections
   - Independent control and configuration

3. The software can access each ALU independently:
```c
// Pseudo-code showing how to access multiple ALUs
snax_alu_0_write_csr(...);  // Access ALU #0
snax_alu_1_write_csr(...);  // Access ALU #1
snax_alu_2_write_csr(...);  // Access ALU #2
```

### Key Configuration for Multiple ALUs

In the config file `snax_multi_alu_one_core_cluster.hjson`:

```hjson
snax_acc_cfg: [
    {
        snax_acc_name: "snax_alu",           // Each ALU instance
        bender_target: ["snax_alu"],
        snax_tcdm_ports: 12,                 // Ports per ALU
        sparse_interconnect_config: [[12, 1]],
        snax_num_rw_csr: 3,                  // R/W CSRs per ALU
        snax_num_ro_csr: 2,                  // Read-only CSRs
        snax_streamer_cfg: {...}
    },
    // ... repeat for additional ALUs
]
```

---

## Part 3: Difference Between Configurations

| Feature | Single ALU | Multi ALU (1 core) | Multi ALU (Multi core) | GEMMX |
|---------|-----------|-------------------|-------------------------|-------|
| Compute Cores | 1 + DMA | 1 + DMA | N + DMA | 1 + DMA |
| ALU Instances | 1 | 3 (configurable) | 3 (1 per core) | 0 (uses GEMMX) |
| Accelerator | snax_alu | snax_alu | snax_alu | snax_gemmX |
| TCDM Ports | 12 | 36 (12×3) | 12 per core | 56 |
| Use Case | Simple operations | Parallel ALU operations on 1 core | Full cluster parallelism | GEMM/Conv operations |

---

## Part 4: Creating Your Own Configuration

To create a custom configuration:

1. Copy one of the base configs:
```bash
cp cfg/snax_gemmx_cluster.hjson cfg/my_custom_cluster.hjson
```

2. Modify the cluster name:
```hjson
cluster: {
    name: "my_custom_cluster",
    bender_target: ["my_custom_cluster", "sparse_interconnect"],
```

3. Adjust accelerator parameters as needed:
```hjson
snax_acc_cfg: [{
    snax_tcdm_ports: 24,  // Increase if needed
    snax_gemmx_mesh_row: 16,  // Larger mesh
    ...
}]
```

4. Build with your custom config:
```bash
pixi run make CFG_OVERRIDE=cfg/my_custom_cluster.hjson rtl-gen
pixi run make CFG_OVERRIDE=cfg/my_custom_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)
```

---

## Part 5: Build and Test Commands

### Standard Build Flow

```bash
cd target/snitch_cluster/

# Step 1: Generate RTL (must do this first when switching configs)
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson rtl-gen

# Step 2: Build hardware simulator
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)

# Step 3: Build software (MUST match hardware config)
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson sw

# Step 4: Create log directory
mkdir -p logs

# Step 5: Run simulation
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-matmul/build/snax-gemmx-matmul.elf

# Step 6 (Optional): Generate traces
pixi run make -j traces
```

### Important Notes

⚠️ **Critical**: When switching between configurations:
- Always run `rtl-gen` to regenerate hardware
- Always rebuild software with matching config
- Delete old `work-vlt` directory if you encounter strange errors:
  ```bash
  rm -rf work-vlt/
  pixi run make CFG_OVERRIDE=cfg/... rtl-gen
  ```

---

## Part 6: Understanding the LRU Config

The SNAX build system uses an LRU (Least Recently Used) config tracking:

```bash
# First build with explicit config
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson rtl-gen

# Subsequent builds remember the config
pixi run make rtl-gen  # Uses snax_gemmx_cluster.hjson automatically

# You can always override
pixi run make CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson rtl-gen
```

The file `cfg/lru.hjson` tracks which config was most recently used.

---

## Summary

✅ **To setup GEMMX**: Use `snax_gemmx_cluster.hjson`
✅ **For multiple ALUs on 1 core**: Use `snax_multi_alu_one_core_cluster.hjson`
✅ **For multiple cores with ALUs**: Use `snax_multi_alu_cluster.hjson` (original)
✅ **Always match hardware and software configs**
✅ **Run `rtl-gen` whenever changing configs**

Both new config files have been created in:
- `/ssd_scratch/Vedant/snax_cluster/target/snitch_cluster/cfg/snax_gemmx_cluster.hjson`
- `/ssd_scratch/Vedant/snax_cluster/target/snitch_cluster/cfg/snax_multi_alu_one_core_cluster.hjson`
