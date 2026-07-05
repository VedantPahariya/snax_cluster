# Configuration Comparison & Architecture Guide

## Architecture Diagrams

### Your Current Setup: snax_alu_cluster.hjson
```
┌─────────────────────────────────────────┐
│         Snitch Cluster                  │
├─────────────────────────────────────────┤
│  Hive 0 (Shared Instruction Cache)      │
│  ┌─────────────────────────────────────┐│
│  │ Core 0 (Compute)  │  Core 1 (DMA)   ││
│  │ ┌───────────────┐                   ││
│  │ │  SNAX-ALU     │                   ││
│  │ │  Accelerator  │                   ││
│  │ │  (1 instance) │                   ││
│  │ └───────────────┘                   ││
│  └─────────────────────────────────────┘│
│              │                           │
│         ┌────┴────┐                      │
│         ↓         ↓                      │
│     TCDM Bank    TCDM Bank              │
│     (32 banks, 128 KB)                  │
└─────────────────────────────────────────┘
```

### Option 1: snax_multi_alu_one_core_cluster.hjson (NEW)
```
┌─────────────────────────────────────────┐
│         Snitch Cluster                  │
├─────────────────────────────────────────┤
│  Hive 0 (Shared Instruction Cache)      │
│  ┌─────────────────────────────────────┐│
│  │ Core 0 (Compute)  │  Core 1 (DMA)   ││
│  │ ┌───────────────┐                   ││
│  │ │  ALU #0       │                   ││
│  │ │  ALU #1       │  (3 independent   ││
│  │ │  ALU #2       │   accelerators)   ││
│  │ └───────────────┘                   ││
│  └─────────────────────────────────────┘│
│              │                           │
│         ┌────┴────┐                      │
│         ↓         ↓                      │
│     TCDM Bank    TCDM Bank              │
│     (32 banks, 128 KB)                  │
└─────────────────────────────────────────┘
```

### Option 2: snax_gemmx_cluster.hjson (NEW)
```
┌──────────────────────────────────────────┐
│         Snitch Cluster                   │
├──────────────────────────────────────────┤
│  Hive 0 (Shared Instruction Cache)       │
│  ┌─────────────────────────────────────┐ │
│  │ Core 0 (Compute)   │  Core 1 (DMA)  │ │
│  │ ┌───────────────┐                   │ │
│  │ │ SNAX GEMMX    │                   │ │
│  │ │ Accelerator   │                   │ │
│  │ │ 8x8 Mesh      │  (Matrix Ops     │ │
│  │ │ w/ Streamer   │   Specialized)   │ │
│  │ └───────────────┘                   │ │
│  └─────────────────────────────────────┘ │
│              │                            │
│         ┌─────┴──────┐                    │
│         ↓            ↓                    │
│     TCDM Bank    TCDM Bank               │
│     (32 banks, 128 KB, 56 ports)        │
└──────────────────────────────────────────┘
```

### Option 3: snax_multi_alu_cluster.hjson (ORIGINAL - Multi-core)
```
┌─────────────────────────────────────────┐
│         Snitch Cluster                  │
├─────────────────────────────────────────┤
│  Hive 0 (Shared Instruction Cache)      │
│  ┌─────────────────────────────────────┐│
│  │ Core 0         │ Core 1              ││
│  │ ┌───────────┐  │ ┌────────────┐    ││
│  │ │  ALU #0   │  │ │  ALU #1    │    ││
│  │ │  ALU #1   │  │ │  ALU #2    │    ││
│  │ │  ALU #2   │  │ │  (DMA Core)│    ││
│  │ └───────────┘  │ └────────────┘    ││
│  └─────────────────────────────────────┘│
│              │                           │
│    ┌─────────┴──────────┐               │
│    ↓                    ↓                │
│  TCDM Banks (32 banks, distributed)    │
└─────────────────────────────────────────┘
```

## Configuration Parameter Breakdown

### GEMMX Configuration (snax_gemmx_cluster.hjson)

**Accelerator Spec:**
```hjson
{
    snax_acc_name: "snax_streamer_gemmX",      // Hardware name
    bender_target: ["snax_gemmX"],              // Build target
    snax_tcdm_ports: 56,                        // Huge! Needs bandwidth

    // GEMMX-specific parameters
    snax_gemmx_mesh_row: 8,                    // 8x8 processing element mesh
    snax_gemmx_tile_size: 8,                   // 8x8 tiles for matrix ops
    snax_gemmx_mesh_col: 8,
    snax_gemmx_serial_c32_d32_width: 2048,     // C matrix 32-bit width
    snax_gemmx_serial_d8_width: 512,           // D matrix 8-bit width
    with_pipeline: true,                        // Pipeline for speed

    // Sparse interconnect config - 4 levels
    sparse_interconnect_config: [
        [8, 1],      // Level 0: 8 ports, 1 connection
        [8, 8],      // Level 1: 8 ports, 8 connections
        [8, 8],      // Level 2: 8 ports, 8 connections
        [32, 8]      // Level 3: 32 ports, 8 connections
    ]
}
```

**Streamer Parameters:**
```hjson
snax_gemmx_streamer_template: {
    data_reader_params: {
        spatial_bounds: [[8], [8]],         // 8x8 spatial domain
        temporal_dim: [6, 3],                // 6 and 3 time dimensions
        num_channel: [8, 8],                 // 8 channels per level
        fifo_depth: [8, 8],                  // 8-deep FIFOs
        tcdm_logic_word_size: [[256, 128, 64], [256, 128, 64]],
        datapath_extensions: [
            {HasTransposer: {row:[8], col:[8], elementWidth:[8]}},
            {HasTransposer: {row:[8], col:[8], elementWidth:[8]}}
        ]
    },
    data_writer_params: {
        spatial_bounds: [[8]],
        temporal_dim: [3],
        num_channel: [8],
        fifo_depth: [1]                     // Shallow output FIFO
    },
    data_reader_writer_params: {
        // For C matrix broadcast and accumulation
        spatial_bounds: [[8, 4], [8, 4]],
        num_channel: [32, 32],               // Wide channels for accumulation
        datapath_extensions: [
            {HasBroadcaster: {inputLength: 256, outputLength: 2048}},
            {}
        ]
    },
    has_transpose: true,                     // Hardware transpose
    has_C_broadcast: true,                   // Broadcast C for efficiency
}
```

### Multi-ALU Configuration (snax_multi_alu_one_core_cluster.hjson)

**Accelerator Spec (repeated 3 times):**
```hjson
[
    {
        snax_acc_name: "snax_alu",                    // Simple ALU
        bender_target: ["snax_alu"],                  // Build target
        snax_tcdm_ports: 12,                          // 12 ports per ALU
        sparse_interconnect_config: [[12, 1]],        // Single level
        snax_num_rw_csr: 3,                          // 3 R/W control registers
        snax_num_ro_csr: 2                           // 2 read-only registers
    },
    { /* same as above for ALU #1 */ },
    { /* same as above for ALU #2 */ }
]
```

**Streamer Parameters (per ALU):**
```hjson
snax_alu_streamer_template: {
    data_reader_params: {
        spatial_bounds: [[4], [4]],         // 4x4 spatial domain
        temporal_dim: [1, 1],                // Simple 1D temporal
        num_channel: [4, 4],                 // 4 channels
        fifo_depth: [8, 8]
    },
    data_writer_params: {
        spatial_bounds: [[4]],
        temporal_dim: [1],
        num_channel: [4],
        fifo_depth: [8]
    },
    snax_library_name: "snax-alu"
}
```

## Key Differences Table

| Aspect | ALU | Multi-ALU (1 core) | GEMMX |
|--------|-----|-------------------|-------|
| **TCDM Ports** | 12 | 12×3 = 36 | 56 |
| **Mesh Size** | N/A | N/A | 8×8 |
| **Operations** | Generic (Add, Mul, etc.) | Generic (×3) | Matrix Ops (GEMM, Conv) |
| **Memory Overhead** | Sparse IC: [[12,1]] | Sparse IC: [[12,1]] | Sparse IC: 4-level |
| **Streamer Complexity** | Simple | Simple | Complex (Reader+Writer+RW) |
| **CSR Per Instance** | 5 (3 RW + 2 RO) | 5 each | 21 (19 RW + 2 RO) |
| **Use Case** | Quick ops | Parallel ALU work | Deep learning, HPC |

## CSR (Control Status Register) Explanation

Each accelerator instance has CSRs for control:

**ALU CSRs (5 total):**
- 3 Read-Write (core controls execution)
- 2 Read-Only (core reads status)

**GEMMX CSRs (21 total):**
- 19 Read-Write (many parameters: mesh config, dataflow, etc.)
- 2 Read-Only (status monitoring)

## Sparse Interconnect Config Explained

The `sparse_interconnect_config` defines hierarchical TCDM access:

**ALU (Simple):**
```hjson
[[12, 1]]
  ↑    ↑
  │    └─ 1 connection point
  └────── 12 ports total
```

**GEMMX (Complex 4-level):**
```hjson
[
    [8, 1],      // Level 0: 8 ports, 1 aggregation
    [8, 8],      // Level 1: 8 ports, 8 aggregations
    [8, 8],      // Level 2: 8 ports, 8 aggregations
    [32, 8]      // Level 3: 32 ports, 8 aggregations
]
// Total: 56 ports hierarchically organized
```

This hierarchical structure allows:
- Multiple readers to different memory locations
- Efficient bandwidth utilization
- Low-latency local access

## Building Custom Configurations

To create a hybrid config (e.g., GEMMX + ALU):

```hjson
snax_acc_cfg: [
    {
        snax_acc_name: "snax_streamer_gemmX",
        bender_target: ["snax_gemmX"],
        // ... GEMMX params ...
    },
    {
        snax_acc_name: "snax_alu",
        bender_target: ["snax_alu"],
        // ... ALU params ...
    }
]
```

This would give you:
- One GEMMX unit (56 ports)
- One ALU unit (12 ports)
- Total: 68 TCDM ports (check if TCDM has enough!)

## Memory Constraints

**Default TCDM: 128 KB with 32 banks**

Each configuration consumes different amounts:
- **ALU**: 12 ports = minimal overhead
- **Multi-ALU (3×)**: 36 ports = moderate overhead
- **GEMMX**: 56 ports = heavy overhead
- **GEMMX + ALU**: 68 ports = very constrained

If you get memory errors, increase TCDM size in cluster config:
```hjson
tcdm: {
    size: 256,  // Increase from 128 KB to 256 KB
    banks: 64   // Increase from 32 to 64 banks
}
```

## Summary

✅ **snax_gemmx_cluster.hjson** = Heavy-duty matrix operations
✅ **snax_multi_alu_one_core_cluster.hjson** = Parallel simple operations
✅ **snax_multi_alu_cluster.hjson** (original) = Multi-core ALU cluster
✅ Mix and match by editing `snax_acc_cfg` array in any config file
✅ Always match hardware config with software build config
