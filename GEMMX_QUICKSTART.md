# Quick Reference: GEMMX & Multi-ALU Setup

## TL;DR - Fastest Path to Success

### Build GEMMX (OpenGEMM Accelerator)
```bash
cd target/snitch_cluster/
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson rtl-gen
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)
pixi run make CFG_OVERRIDE=cfg/snax_gemmx_cluster.hjson sw
mkdir -p logs
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-matmul/build/snax-gemmx-matmul.elf
```

### Build Multi-ALU (1 Core, 3 ALU Instances)
```bash
cd target/snitch_cluster/
pixi run make CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson rtl-gen
pixi run make CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)
pixi run make CFG_OVERRIDE=cfg/snax_multi_alu_one_core_cluster.hjson sw
mkdir -p logs
./bin/snitch_cluster.vlt ./sw/apps/snax-multi-alu/build/snax-multi-alu.elf
```

## Answer to Your Questions

### Q: Can multiple cores work with ONE ALU accelerator?
**A: Yes** - But you configure it differently:
- **One core, multiple ALU instances**: Use `snax_multi_alu_one_core_cluster.hjson`
- **Multiple cores, each with one ALU**: Use `snax_multi_alu_cluster.hjson` (original)
- Each core independently controls its ALU instance via separate CSRs

### Q: How to have multiple cores with one ALU?
Each core needs its own instance of the accelerator in the `snax_acc_cfg` array. Cores in different hives don't share accelerators - each must be explicitly added.

## Config Files Created

1. **snax_gemmx_cluster.hjson** - GEMMX/OpenGEMM accelerator with 8x8 mesh
2. **snax_multi_alu_one_core_cluster.hjson** - Single core with 3 ALU instances

## Key Concepts

### snax_acc_cfg Array
This is an **ARRAY** of accelerators, not a single object. Each entry = one accelerator instance.

```hjson
snax_acc_cfg: [
    { snax_acc_name: "snax_alu", ... },  // Accelerator 0
    { snax_acc_name: "snax_alu", ... },  // Accelerator 1
    { snax_acc_name: "snax_alu", ... },  // Accelerator 2
]
```

### TCDM Ports
- ALU: 12 ports per instance
- GEMMX: 56 ports (large mesh needs more bandwidth)

### Important Files in Configs
- `bender_target`: Which hardware module to build (snax_alu vs snax_gemmX)
- `snax_acc_cfg`: Accelerator configuration (number of instances, parameters)
- `snax_streamer_cfg`: Data streaming configuration (spatial/temporal bounds)

## Workflow Checklist

- [ ] Choose config (snax_gemmx_cluster.hjson or snax_multi_alu_one_core_cluster.hjson)
- [ ] Run `rtl-gen` to generate RTL from config
- [ ] Build hardware with `bin/snitch_cluster.vlt`
- [ ] Build software matching the same config
- [ ] Create logs directory
- [ ] Run simulation with .elf file
- [ ] (Optional) View traces with `make -j traces`

## Troubleshooting

**Error: Package not found (snax_alu_cluster_pkg)**
→ Re-run `rtl-gen`: `pixi run make CFG_OVERRIDE=... rtl-gen`

**Error: Software won't build**
→ Ensure you're using the SAME config for both hardware and software

**Simulation hangs or crashes**
→ Delete work-vlt directory and rebuild:
```bash
rm -rf work-vlt/
pixi run make CFG_OVERRIDE=... rtl-gen
pixi run make CFG_OVERRIDE=... bin/snitch_cluster.vlt -j$(nproc)
```

**Bus error or strange crashes**
→ Corrupted Pixi cache - clean and rebuild:
```bash
rm -rf .pixi/
pixi install
pixi run make CFG_OVERRIDE=... rtl-gen
```

## Config Location
Both new configs are in:
```
/ssd_scratch/Vedant/snax_cluster/target/snitch_cluster/cfg/
```

## Next Steps
1. Try GEMMX build first (most straightforward)
2. Then try Multi-ALU to understand accelerator arrays
3. Modify parameters in configs based on your needs
