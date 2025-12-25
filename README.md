# hip-onion-vanity

A high-performance, cryptographically secure Tor v3 vanity address generator optimized specifically for AMD GPUs using the ROCm/HIP framework.

This miner utilizes the Montgomery Ladder (5M + 2A) algorithm and RDNA 2/3 optimized math to achieve maximum iteration speeds. It implements a "Random Base + Linear Offset" search strategy with **automatic CSPRNG re-seeding** upon every match, ensuring generated identities are mathematically unlinkable.

## Features

- **AMD Native**: Built on C++ HIP for direct hardware access (no OpenCL overhead).
- **Cryptographically Secure**: Uses System Entropy (`std::random_device`) to generate base keys and re-seeds immediately after finding a match to prevent key correlation.
- **Algo Optimized**: Uses "Y-Only" differential addition to reduce calculations per key by ~35%.
- **Multi-Target**: Search for multiple prefixes simultaneously without speed loss.
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

## Output & Key Usage

When a match is found, the miner outputs the **Raw Scalar** (Private Key) and the Onion URL.

Example `found.txt` content:

```text
==================================================
Onion URL:   testyq5bmmrcz4plmtfel6oyrvis3wksm7yrdvr6z4ghrp6eyauvtwid.onion
Private Key: e2b6933bc2bbd9248c4ea344b4d9465f44eb4165c7c9c652a985cb8761e9a6ed
Public Key:  99253c43a163222cf1eb64ca45f9d88d512dd95267f111d63ecf0c78bfc4c029
NOTE: Private Key is a RAW SCALAR (not a seed).
==================================================
```

### How to use the keys

The output is a **Raw Ed25519 Scalar**. To use this with Tor:

1. You must convert the 64-character Hex string into a binary file named `hs_ed25519_secret_key`.
2. The file must contain the 32-byte scalar followed by 32 bytes of random extension data (and the Tor header).
3. Place this file in your Hidden Service directory (e.g., `/var/lib/tor/hidden_service/`).

## Disclaimer

This tool generates valid Ed25519 private scalars. While it implements CSPRNG re-seeding to ensure unlinkability between found keys, users are responsible for securely handling the generated private keys. Use for educational and security research purposes only.
