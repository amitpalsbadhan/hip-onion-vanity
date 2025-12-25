# hip-onion-vanity

A high-performance, custom-built Tor v3 vanity address generator optimized specifically for AMD GPUs using the ROCm/HIP framework.

This miner utilizes the Montgomery Ladder (5M + 2A) algorithm and RDNA 2/3 optimized math to achieve maximum iteration speeds on cards like the Radeon 6800 XT, 7900 XTX, and similar architectures.

## Features

- **AMD Native**: Built on C++ HIP for direct hardware access (no OpenCL overhead).
- **Algo Optimized**: Uses "Y-Only" differential addition to reduce calculations per key by ~35%.
- **Multi-Target**: Search for multiple prefixes simultaneously without speed loss.
- **Zero-Copy**: GPU only interrupts the CPU when a valid match is found.
- **RDNA Optimized**: Uses C++ intrinsics that compile to efficient `v_add_co_ci_u32` instruction chains.

## Performance

Achieves **450 Million keys/sec** on a single AMD Radeon 6800 XT.

## Prerequisites

- **OS**: Windows 10/11 or Linux.
- **Hardware**: AMD GPU (RDNA 2 or newer recommended).
- **Software**: AMD ROCm / HIP SDK installed and added to PATH.

## Compilation

Open your terminal (PowerShell or CMD on Windows) in the project folder and run:

```bash
hipcc -O3 --offload-arch=native -Iinclude src/main.cpp src/kernels.hip src/sha3.cpp src/utils.cpp -o vanity_miner.exe
```

- `-O3`: Maximum optimization level.
- `--offload-arch=native`: Automatically detects your GPU (e.g., gfx1030 for 6800 XT) and generates specialized assembly.
- `-Iinclude`: Includes the header files directory.

## Usage

### Basic Search

Search for a single prefix (e.g., "test"):

```bash
./vanity_miner.exe -p test
```

### Multi-Prefix Search

Search for multiple prefixes at the same time (e.g., "alex", "shop", "buy"):

```bash
./vanity_miner.exe -p alex -p shop -p buy
```

### Time-Limited Search

Run the miner for a specific number of seconds (e.g., 60 seconds) and then stop:

```bash
./vanity_miner.exe -p test -t 60
```

## Output

- **Console**: Displays real-time speed (Mops/s) and total keys searched.
- **File**: When a match is found, it is appended to `found.txt` (or `prefix.txt`) in the same directory.

Example `found.txt` content:

```text
==================================================
Onion URL:   test...d2id.onion
Private Key: 1a2b3c4d...
Public Key:  5e6f7g8h...
NOTE: Private Key is a RAW SCALAR (not a seed).
==================================================
```

## Disclaimer

This tool is a high-performance Proof of Concept. It generates the Public Key (Y-coordinate) upon finding a match. For a fully functional onion service, you would need to extend the code to save the random seed used to generate the private key. Use for educational and security research purposes only.
