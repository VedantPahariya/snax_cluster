# SNAX Cluster Custom Setup & Troubleshooting Guide

This document captures the complete sequence of commands, the errors encountered while following the official `introduction.md`, and the exact fixes required to successfully run the `snax-alu` accelerator on a system without `root`/`sudo` or `docker` privileges.

## Why the Official Guide Fails out of the Box
The official documentation assumes you are either using their pre-built Docker container or have `sudo` access to install system-wide dependencies like `verilator` and `python3` packages. On a restricted environment (like shared computing nodes), commands like `docker run` or `apt-get install` will fail. 

To solve this, we used **Pixi**, a package manager defined in the repository's `pixi.toml`, to locally provision all the required tools (`verilator`, `bender`, `clang`, `make`, etc.) without needing administrative access.

---

## Complete Working Command Sequence

Run these commands from the root of the `snax_cluster` repository to seamlessly build and run the test harness:

```bash
# 1. Update submodules
git submodule update --init --recursive

# 2. Install Pixi locally (if not already installed)
curl -fsSL https://pixi.sh/install.sh | bash
export PATH="$HOME/.pixi/bin:$PATH"

# 3. Configure Pixi to use a scratch directory to clear storage quota limits
export PIXI_CACHE_DIR=/ssd_scratch/Vedant/.pixi_cache
export RATTLER_CACHE_DIR=/ssd_scratch/Vedant/.rattler_cache

# 4. Install the environment (Verilator, Bender, Clang, Python deps)
pixi install

# Choosing config by defing a variable
CONFIG=snax_KUL_cluster.hjson

# 5. Generate RTL and Bender Targets
pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/${CONFIG} rtl-gen

# 6. Build the Hardware Emulator (Verilator)
pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/${CONFIG} bin/snitch_cluster.vlt -j$(nproc)

# 7. Build the Software Payload (sequentially to avoid memory/race conditions)
pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/${CONFIG} sw

# 8. Run the Emulator
./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-alu/build/snax-alu.elf
```

The software command used above uses `rtl-generic`, which cannot compile DNN app because it lacks SSR declarations. Run the following command:

```bash
pixi run make -C target/snitch_cluster \
  CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson \
  SELECT_RUNTIME=rtl \
  SELECT_TOOLCHAIN=llvm-snitch \
  sw
```

This command is logically correct for our DNN app, but cannot run here until /tools/riscv-llvm exists. This is Snitch LLVM toolchain which is hardcoded inside `snax_cluster/target/snitch_cluster/sw/toolchain.mk`.

---

## Working `snax_KUL_cluster` + DNN Smoke Flow

The missing Snitch LLVM toolchain has been installed locally at:

```bash
/ssd_scratch/Vedant/tools/riscv-llvm
```

The toolchain came from the PULP/Snitch LLVM release:

```text
15.0.0-snitch-0.5.0
riscv32-snitch-llvm-ubuntu2204-15.0.0-snitch-0.5.0.tar.gz
```

`target/snitch_cluster/sw/toolchain.mk` now uses `LLVM_SNITCH_ROOT`, so the path is configurable instead of hardcoded to `/tools/riscv-llvm`.

Use this exact sequence from the repository root:

```bash
cd /ssd_scratch/Vedant/snax_cluster

export PIXI_CACHE_DIR=/ssd_scratch/Vedant/.pixi_cache
export RATTLER_CACHE_DIR=/ssd_scratch/Vedant/.rattler_cache
export LLVM_SNITCH_ROOT=/ssd_scratch/Vedant/tools/riscv-llvm

pixi install

pixi run make -C target/snitch_cluster \
  CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson \
  rtl-gen -j$(nproc)

pixi run ./util/snaxgen/snaxgen.py \
  --cfg_path=$PWD/target/snitch_cluster/cfg/snax_KUL_cluster.hjson \
  --get_bender_targets > target/snitch_cluster/generated/bender_targets.tmp

pixi run make -C target/snitch_cluster \
  CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson \
  bin/snitch_cluster.vlt -j$(nproc)

pixi run make -C target/snitch_cluster \
  CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson \
  SELECT_RUNTIME=rtl \
  SELECT_TOOLCHAIN=llvm-snitch \
  LLVM_SNITCH_ROOT=$LLVM_SNITCH_ROOT \
  sw

cd target/snitch_cluster
mkdir -p logs
./bin/snitch_cluster.vlt ./sw/apps/snax-gemmx-matmul/build/snax-gemmx-matmul.elf
./bin/snitch_cluster.vlt ./sw/apps/snax-dnn-inference/build/snax-dnn-inference.elf
```

Verified outputs:

```text
SNAX GEMM Matmul: PASS, Error: 0 . bypassSIMD = 1 .
SNAX DNN Inference RV32IMA smoke: PASS, checksum: 0x15124658
```

Important caveat: `snax_KUL_cluster.hjson` sets both cores to `isa: "rv32ima"`. That means the CPU cores do not support the RISC-V `F` or `D` floating-point extensions. Any DNN code path that executes `float`/`double` CPU arithmetic will trap in RTL with an illegal instruction. The current DNN app therefore runs an integer-only smoke path on this config. A real DNN inference path on this cluster must either be quantized/integer-only or offload arithmetic to the GEMMX/SNAX datapath instead of executing FP32/FP64 on the Snitch core.

The upstream `sw/apps/dnn/*` tests are PyTorch-dependent. They are now opt-in with:

```bash
BUILD_DNN_TESTS=1
```

Leave this unset for the normal KUL software build unless PyTorch is installed in the Pixi environment.

---

## Errors Faced & How They Were Resolved

### Error 1: Missing System Tools (`verilator: not found`, `mako` module error)
- **Problem**: When running the raw `make ... rtl-gen` command, the system lacks `verilator`, the Python library `mako`, and `bender`.
- **Attempt**: Tried using `sudo apt-get` and `docker`. Both failed because of permission restrictions.
- **Solution**: We looked into `pixi.toml` and realized the developers packaged the required toolchain using Pixi. We downloaded and installed `pixi` locally in user space, and used `pixi run make` so the environment had access to all properly versioned toolchains.

### Error 2: Disk Quota Exceeded during Pixi Install
- **Problem**: `Quota exceeded (os error 122)` in the `/home2/vedu.p/` directory when Pixi tried to unpack heavy dependencies like `llvm` and `verilator`.
- **Solution**: Set the caching environment variables (`PIXI_CACHE_DIR` and `RATTLER_CACHE_DIR`) to point to the `/ssd_scratch/Vedant/` disk instead of the home directory, fully bypassing space limitations.

### Error 3: Transient `Bus Error (core dumped)` in Clang / Make
- **Problem**: While building the software, parallel compilation (`-j`) combined with a slightly corrupted Conda cache resulted in strange behavior where `clang-format` and `clang` threw Bus Errors (memory access violations) out of nowhere.
- **Solution**: It was determined that the binaries unpacked during the "Quota exceeded" phase were corrupted. We safely eradicated the previous `.pixi/` and caches, forced a completely bold `pixi install`, and rebuilt the software gracefully (without `-j`) to prevent any I/O collisions. (Optionally overriding `CLANG_FORMAT=true` also helped sidestep bad binary faults initially).

### Error 4: Verilator Cannot Find Pakcage (`snax_alu_cluster_pkg`)
- **Problem**: When compiling the HW using Verilator, it threw `Package/class 'snax_alu_cluster_pkg' not found`.
- **Solution**: This occurred because `bender_targets.tmp` was empty due to a previous half-failed `rtl-gen` execution. By deleting `generated/bender_targets.tmp` and re-running the `rtl-gen` target properly inside the Pixi environment, the system populated the target definitions, dynamically mapping the ALU correctly.

# SNAX Build System Pipeline

Here we explains what happens behind each command in the setup guides and why each step is necessary.

## Overview of the Build Pipeline

The SNAX build system is a **three-stage pipeline**:

1. **rtl-gen** -  Define hardware accelerators and generate RTL + C headers from JSON configs (./target/snitch_cluster/cfg/*.hjson)
   - Uses wrapper generator (snaxgen.py) to create streamers hardware (data routing), CSR managers(control registers), wrappers
   - Creates `bender_targets.tmp` listing which RTL modules to compile
   - Software headers (CSR addresses) go here
   - **Critical:** Must run FIRST when changing CFG_OVERRIDE

2. **bin/snitch_cluster.vlt** - Compile RTL into a Verilator simulator executable
   - Verilates RTL (SystemVerilog → C++)
   - Links test bench + FESVR library
   - Produces simulator executable
   - Requires generated files from rtl-gen

3. **sw** - Build RISC-V software
   - Uses headers generated by `rtl-gen` to know CSR addresses
   - Generates C headers using clustergen + reggen
   - Builds runtime, accelerator drivers, test apps
   - Headers contain CSR addresses - must match hardware config

## Config System: LRU Mechanism

- `cfg/lru.hjson` is symlink to most recently used config
- First command sets it: `make CFG_OVERRIDE=cfg/snax_alu_cluster.hjson rtl-gen`
- Subsequent commands use it automatically (no need to repeat CFG_OVERRIDE)
- Prevents config mismatches between hardware and software

## Key Files

- `/target/snitch_cluster/Makefile` - Main build
- `/target/snitch_cluster/sw/Makefile` - Software build (recursive)
- `/target/snitch_cluster/cfg/*.hjson` - Configuration files
- `/generated/` - Output of rtl-gen (RTL + headers)

## Command-by-Command Breakdown

### 1. `git submodule update --init --recursive`
**What it does:**
- Initializes and updates all git submodules that the SNAX project depends on
- SNAX uses external repositories for components like: Verilator, Bender (dependency manager), iDMA (DMA controller), etc.

**Why needed:**
- SNAX relies on many external components to function. These are stored as submodules rather than in the main repo to reduce clutter and enable independent versioning
- Without this, the required dependencies won't exist when the build system tries to use them

**Makefile location:** Not in the Makefile, it's a pre-build step required by Pixi

---

### 2. `curl -fsSL https://pixi.sh/install.sh | bash` + `export PATH="$HOME/.pixi/bin:$PATH"`
**What it does:**
- Downloads and installs Pixi (a package manager similar to Conda but faster and lighter)
- Adds Pixi to the system PATH so it can be invoked from anywhere

**Why needed:**
- The SNAX project defines all its dependencies (Verilator, Clang, Python packages, etc.) in `pixi.toml`
- Pixi installs these locally in userspace without needing `sudo` or root access
- This bypasses the restrictions of shared computing environments

**Makefile location:** Not in the Makefile, it's the environment setup

---

### 3. `export PIXI_CACHE_DIR=/ssd_scratch/Vedant/.pixi_cache` + `export RATTLER_CACHE_DIR=/ssd_scratch/Vedant/.rattler_cache`
**What it does:**
- Configures where Pixi stores its package cache and dependencies
- Points to `/ssd_scratch/` (a larger disk) instead of the home directory

**Why needed:**
- The home directory on shared clusters often has strict quota limits
- Large tools like Verilator (~200MB+) and LLVM (~500MB+) don't fit in typical home quotas
- By pointing to `/ssd_scratch/`, we use a larger disk without hitting quota errors

**Makefile location:** Not in the Makefile, it's an environment configuration

---

### 4. `pixi install`
**What it does:**
- Reads `pixi.toml` from the repo root
- Downloads and installs all defined dependencies into a local Pixi environment
- Creates the `.pixi/envs/default/` directory with all tools

**Why needed:**
- Ensures all required tools are available for subsequent build steps
- Tools installed: verilator (hardware simulator), bender (RTL dependency manager), clang (C/C++ compiler), Python packages, etc.

**Makefile location:** Not a Makefile target, but Pixi runs commands via `pixi run make ...`

---

### 5. `pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson rtl-gen`

This is the first actual Makefile target. Let's break it down:

#### `pixi run make`
- Runs the `make` command within the Pixi environment (so it has access to all installed tools)

#### `-C target/snitch_cluster`
- Changes directory to `target/snitch_cluster/` before running make

#### `CFG_OVERRIDE=cfg/snax_alu_cluster.hjson`
- Tells the Makefile to use this specific configuration file
- The configuration file defines accelerator types, matrix dimensions, memory sizes, etc.

#### `rtl-gen` (the actual target)
**Location:** Makefile line 155

```makefile
rtl-gen: $(GENERATED_DIR)/bender_targets.tmp rtl-snax-gen rtl-cluster-gen idma-rtl-gen
```

**What it does (3 sub-stages):**

**a) `$(GENERATED_DIR)/bender_targets.tmp` (line 140)**
```makefile
$(GENERATED_DIR)/bender_targets.tmp: | $(GENERATED_DIR)
    ${WRAPPER_GEN} --cfg_path="$(CFG_FILE)" --get_bender_targets > $(GENERATED_DIR)/bender_targets.tmp
```
- Runs the wrapper generator with the config file
- Generates a list of Bender targets (which RTL modules to build)
- Outputs to `bender_targets.tmp`

**b) `rtl-snax-gen` (line 100)**
```makefile
rtl-snax-gen: | $(GENERATED_DIR)
    ${WRAPPER_GEN} --cfg_path="$(CFG_FILE)" \
        --tpl_path="${SNAX_TPL_PATH}" \
        --test_path="${SNAX_TEST_PATH}" \
        --chisel_path="${SNAX_CHISEL_PATH}" \
        --bypass_accgen="${BYPASS_ACCGEN}" \
        --disable_header_gen="${DISABLE_HEADER_GEN}" \
        --gen_path="${GENERATED_DIR}/"
```
- Runs the wrapper generator (snaxgen.py) with full options
- **Generates:**
  - Streamer RTL (data routing hardware)
  - CSR Manager RTL (control registers)
  - Wrappers connecting accelerators to the cluster
  - Test harness (verification framework)
  - Software header files (CSR address maps)

**c) `rtl-cluster-gen` (line 133)**
```makefile
rtl-cluster-gen: | $(GENERATED_DIR)
    $(CLUSTER_GEN) -c ${CFG_OVERRIDE} -o $(GENERATED_DIR) --wrapper
```
- Runs clustergen.py
- Generates the top-level cluster wrapper RTL

**d) `idma-rtl-gen` (line 149)**
- Generates iDMA (intelligent Direct Memory Access) RTL for data transfers

**Why `rtl-gen` is critical:**
- Creates all hardware adapters and wrappers needed for your specific config
- Generates software headers so the executable knows where CSRs are located
- Must be re-run whenever you change the config file

---

### 6. `pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson bin/snitch_cluster.vlt -j$(nproc)`

#### `bin/snitch_cluster.vlt` (the target)
**Location:** Makefile line 365

```makefile
bin/snitch_cluster.vlt: $(VLT_AR) $(VLT_COBJ) ${VLT_BUILDDIR}/lib/libfesvr.a
    mkdir -p $(dir $@)
    $(CXX) $(LDFLAGS) $(VLT_CXXSTD_FLAGS) -L ${VLT_BUILDDIR}/lib -o $@ $(VLT_COBJ) $(VLT_AR) -lfesvr
```

**What it does:**

**a) Build dependencies:**

**`$(VLT_AR)` = `${VLT_BUILDDIR}/Vtestharness__ALL.a` (line 166)**
- Verilates all RTL (Systemverilog → C++ code)
- Compiles C++ into a static library archive (`.a` file)
- This is the simulation kernel

**`$(VLT_COBJ)` (line 235-249)**
- C/C++ test bench objects that hook into the Verilated model
- Includes bootrom, IPC, common libraries, simulation framework

**`${VLT_BUILDDIR}/lib/libfesvr.a`**
- FESVR (Frontend Server) library
- Provides instruction trace support and system call handling for RISC-V programs

**b) Link everything together:**
- Combines the Verilated RTL (C++), test bench code, and FESVR library
- Produces `bin/snitch_cluster.vlt` - an executable that simulates the hardware

#### `-j$(nproc)`
- Parallel compilation using all CPU cores
- `$(nproc)` returns number of processors

**Why this step matters:**
- You cannot run any tests until the simulator is built
- Vectorizes/optimizes hardware simulation for speed

---

### 7. `pixi run make -C target/snitch_cluster CFG_OVERRIDE=cfg/snax_alu_cluster.hjson sw`

**Location:** Makefile line 305

```makefile
sw: $(TARGET_C_HDRS)
    $(MAKE) -C $(SW_DIR) CFG_OVERRIDE=$(CFG_OVERRIDE)
```

**What it does:**

**a) Generate C headers from config:**

**Software headers created (line 296-301):**
```makefile
TARGET_C_HDRS = $(addprefix $(TARGET_C_HDRS_DIR)/,$(CLUSTER_GEN_HEADERS) $(REGGEN_HEADERS))
```

- `snitch_cluster_cfg.h` - Cluster configuration (memory layout, core count, etc.) - generated by clustergen.py
- `snitch_cluster_addrmap.h` - Memory address map
- `snitch_cluster_peripheral.h` - CSR register definitions - generated by reggen

**These headers tell software:**
- Where CSRs are in memory (CSR addresses)
- What the cluster looks like (how many ALUs, streamer config)
- How to talk to accelerators

**Example of what `snitch_cluster_cfg.h` contains:**
```c
#define SNITCH_CLUSTER_TCDM_START_ADDR 0x10000000
#define SNITCH_CLUSTER_TCDM_SIZE 65536
#define SNAX_ALU_CSR_ADDR 0x304000
#define SNAX_STREAMER_ADDR 0x300000
```

**b) Compile software:**
```makefile
$(MAKE) -C $(SW_DIR) CFG_OVERRIDE=$(CFG_OVERRIDE)
```

**Location:** `/target/snitch_cluster/sw/Makefile` (lines 18-61)
- Recursively builds math libraries, runtime, accelerator drivers, and test apps
- Each app is compiled to RISC-V ELF format

**Why matching config is critical:**
- If hardware has CSRs at address X but software thinks they're at Y, the app crashes
- All hardware and software must be built from the **same config file**

---

### 8. `./target/snitch_cluster/bin/snitch_cluster.vlt ./target/snitch_cluster/sw/apps/snax-alu/build/snax-alu.elf`

**What it does:**
- Runs the simulator with a specific test program
- The simulator:
  - Executes the RISC-V ELF instructions from the program
  - Responds to accelerator CSR writes
  - Simulates memory and data transfers
  - Produces execution traces

**Output:**
- Text output showing program results
- VCD trace files (in `logs/`) showing signal activity

---

## Config File System: The LRU Mechanism

**Location:** Makefile lines 173-284

```makefile
DEFAULT_CFG = cfg/default.hjson
CFG         = cfg/lru.hjson
```

**How it works:**

1. **`cfg/lru.hjson` is a symlink** pointing to the most recently used config
2. When you run with `CFG_OVERRIDE=cfg/snax_alu_cluster.hjson`:
   - The Makefile symlink `cfg/lru.hjson` → `cfg/snax_alu_cluster.hjson`
   - Subsequent `make` commands automatically use this config
   - You don't need to keep specifying `CFG_OVERRIDE`

3. **Why this matters:**
   - Prevents accidental config mismatches
   - If hardware built with Config A but software built with Config B, nothing works
   - LRU tracking ensures consistency

---

## Why Both Guides Say "Almost the Same" Commands

The two guides show the same pipeline but for different reasons:

| Step | CUSTOM_SETUP_GUIDE | SNAX_SETUP_GUIDE |
|------|-----------------|-----------------|
| Purpose | Debugging restricted environments | Different accelerator configs |
| Commands | Same build steps, different reasons | Same build steps, different configs |
| Key difference | Focuses on quota issues and Pixi workarounds | Focuses on using different accelerator types (GEMMX vs ALU) |

**Both follow the same 3-stage pipeline:**
1. RTL generation (`rtl-gen`)
2. Hardware compilation (`bin/snitch_cluster.vlt`)
3. Software build (`sw`)

The **only difference** is which config file you use:
- `cfg/snax_alu_cluster.hjson` - for ALU accelerator
- `cfg/snax_gemmx_cluster.hjson` - for GEMMX (matrix multiply) accelerator

---

## The Build Dependency Chain

```
rtl-gen
  ├── bender_targets.tmp (what RTL modules exist)
  ├── rtl-snax-gen (streamer + CSR manager + wrappers)
  ├── rtl-cluster-gen (cluster wrapper)
  └── idma-rtl-gen (DMA RTL)
      → Generates: /generated/*.sv and C headers

bin/snitch_cluster.vlt
  ├── $(VLT_AR)
  │   ├── Verilate RTL → C++
  │   └── Compile C++ → .a archive
  ├── $(VLT_COBJ)
  │   ├── Compile testbench C++ files
  │   └── Compile Verilator runtime
  └── libfesvr.a (instruction trace)
      → Produces: bin/snitch_cluster.vlt (the simulator executable)

sw
  ├── Generate C headers (from rtl-gen outputs)
  ├── Build runtime (RISC-V code running on simulated cores)
  ├── Build accelerator drivers (snax-alu, gemmx, etc.)
  └── Build test apps (snax-alu.elf, snax-gemmx-matmul.elf, etc.)
```

**Critical constraint:** You MUST run `rtl-gen` before `bin/snitch_cluster.vlt` and before `sw`, because:
- RTL generation creates the headers that software needs
- Hardware simulator needs generated RTL
- Both depend on generated JSON/header files

---

## Common Errors Explained

**Error: "snax_alu_cluster_pkg not found"**
- Cause: `rtl-gen` didn't fully complete or `bender_targets.tmp` is empty
- Fix: Delete `generated/` and re-run `rtl-gen`

**Error: "CFG_OVERRIDE mismatch"**
- Cause: Built hardware with Config A, software with Config B
- Fix: Use same CFG_OVERRIDE for all commands in a build session

**Error: "Bus Error" during `make sw`**
- Cause: Corrupted Pixi cache (from quota errors)
- Fix: Clear `.pixi/` and caches, re-run `pixi install` from scratch

---

## Summary Table

| Command | Stage | Purpose | Why Needed |
|---------|-------|---------|-----------|
| `rtl-gen` | 1 | Generate RTL & C headers for config | Software needs header files; RTL simulator needs generated modules |
| `bin/snitch_cluster.vlt` | 2 | Build Verilator simulator | Need executable to run tests |
| `sw` | 3 | Build RISC-V software | Need ELF files to load into simulator |
| `./bin/snitch_cluster.vlt app.elf` | 4 | Run test | Execute test program in simulator |

## Config Choices

- `snax_alu_cluster.hjson` - Simple ALU accelerators
- `snax_gemmx_cluster.hjson` - Matrix multiply (ML workloads)
- `snax_multi_alu_one_core_cluster.hjson` - Multiple ALUs on single core
