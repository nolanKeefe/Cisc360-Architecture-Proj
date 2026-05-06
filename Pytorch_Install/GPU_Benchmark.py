"""
gpu_benchmark.py
PyTorch GPU Benchmark Script
Compares speed across containers by running standardized GPU compute tests.
Usage: python3 gpu_benchmark.py
"""

import time
import torch
import platform
import sys

# ─────────────────────────────────────────────
# CONFIGURATION — adjust these if needed
# ─────────────────────────────────────────────
MATRIX_SIZE    = 8192      # Size of matrices for matmul test (N x N)
MATMUL_REPEATS = 50        # How many times to repeat the matmul test
CONV_BATCH     = 64        # Batch size for convolution test
CONV_REPEATS   = 50        # How many times to repeat the conv test
WARMUP_RUNS    = 5         # Warm-up runs before timing (GPU needs to "wake up")
# ─────────────────────────────────────────────


def print_header(title):
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def print_section(title):
    print(f"\n── {title} " + "─" * (55 - len(title)))


def system_info():
    print_header("SYSTEM & ENVIRONMENT INFO")
    print(f"  Python        : {sys.version.split()[0]}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n  ⚠ WARNING: CUDA is not available!")
        print("  Make sure you launched Apptainer with --nv")
        print("  and that you are on a GPU compute node.")
        sys.exit(1)

    print(f"  CUDA version  : {torch.version.cuda}")
    print(f"  cuDNN version : {torch.backends.cudnn.version()}")
    print(f"  GPU count     : {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(f"  GPU {i}         : {props.name}  ({total_mem_gb:.1f} GB VRAM)")

    print(f"  Platform      : {platform.platform()}")


def timer(func, *args, repeats=10, warmup=WARMUP_RUNS, device=None, **kwargs):
    """
    Runs a function multiple times and returns timing statistics.
    Does warm-up runs first so the GPU is fully active before measuring.
    """
    if device is None:
        device = torch.device("cuda")

    # Warm-up: run without timing so GPU reaches steady state
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize(device)  # Wait for all GPU ops to finish

    # Timed runs
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func(*args)
        torch.cuda.synchronize(device)  # Crucial: ensures GPU is done before stopping timer
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to milliseconds

    avg  = sum(times) / len(times)
    best = min(times)
    worst = max(times)
    return avg, best, worst


# ─────────────────────────────────────────────
# TEST 1: Matrix Multiplication (FP32)
# ─────────────────────────────────────────────
def run_matmul_fp32(device):
    print_section("Test 1: Matrix Multiplication — FP32")
    print(f"  Matrix size : {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"  Repeats     : {MATMUL_REPEATS}  (+ {WARMUP_RUNS} warm-up)")

    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float32)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float32)

    def matmul():
        return torch.matmul(A, B)

    avg, best, worst = timer(matmul, repeats=MATMUL_REPEATS, device=device)

    # TFLOPS = 2 * N^3 operations / time in seconds / 1e12
    tflops = (2 * MATRIX_SIZE ** 3) / (avg / 1000) / 1e12

    print(f"\n  Avg time     : {avg:.2f} ms")
    print(f"  Best time    : {best:.2f} ms")
    print(f"  Worst time   : {worst:.2f} ms")
    print(f"  Throughput   : {tflops:.2f} TFLOPS")
    return avg, tflops


# ─────────────────────────────────────────────
# TEST 2: Matrix Multiplication (FP16 / half precision)
# A100s have Tensor Cores that massively accelerate FP16
# ─────────────────────────────────────────────
def run_matmul_fp16(device):
    print_section("Test 2: Matrix Multiplication — FP16 (Tensor Cores)")
    print(f"  Matrix size : {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"  Repeats     : {MATMUL_REPEATS}  (+ {WARMUP_RUNS} warm-up)")

    A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)
    B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device, dtype=torch.float16)

    def matmul():
        return torch.matmul(A, B)

    avg, best, worst = timer(matmul, repeats=MATMUL_REPEATS, device=device)
    tflops = (2 * MATRIX_SIZE ** 3) / (avg / 1000) / 1e12

    print(f"\n  Avg time     : {avg:.2f} ms")
    print(f"  Best time    : {best:.2f} ms")
    print(f"  Worst time   : {worst:.2f} ms")
    print(f"  Throughput   : {tflops:.2f} TFLOPS")
    return avg, tflops


# ─────────────────────────────────────────────
# TEST 3: 2D Convolution (simulates a CNN layer)
# ─────────────────────────────────────────────
def run_conv2d(device):
    print_section("Test 3: 2D Convolution (CNN-style workload)")
    print(f"  Input  : batch={CONV_BATCH}, channels=256, size=56x56")
    print(f"  Filter : 512 filters, 3x3 kernel")
    print(f"  Repeats: {CONV_REPEATS}  (+ {WARMUP_RUNS} warm-up)")

    conv = torch.nn.Conv2d(
        in_channels=256,
        out_channels=512,
        kernel_size=3,
        padding=1
    ).to(device)

    x = torch.randn(CONV_BATCH, 256, 56, 56, device=device)

    def forward_pass():
        with torch.no_grad():
            return conv(x)

    avg, best, worst = timer(forward_pass, repeats=CONV_REPEATS, device=device)

    print(f"\n  Avg time     : {avg:.2f} ms")
    print(f"  Best time    : {best:.2f} ms")
    print(f"  Worst time   : {worst:.2f} ms")
    return avg


# ─────────────────────────────────────────────
# TEST 4: Training Step (forward + backward pass)
# Most realistic test for actual ML workloads
# ─────────────────────────────────────────────
def run_training_step(device):
    print_section("Test 4: Training Step — Forward + Backward Pass")
    print(f"  Model  : ResNet-style 4-layer conv network")
    print(f"  Batch  : {CONV_BATCH} x 3 x 224 x 224  (ImageNet-sized input)")
    print(f"  Repeats: {CONV_REPEATS}  (+ {WARMUP_RUNS} warm-up)")

    # Simple 4-layer conv model (representative of real training)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3,   64,  kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64,  128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(256, 512, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(512, 1000)
    ).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    x      = torch.randn(CONV_BATCH, 3, 224, 224, device=device)
    labels = torch.randint(0, 1000, (CONV_BATCH,), device=device)

    def training_step():
        optimizer.zero_grad()
        output = model(x)
        loss   = criterion(output, labels)
        loss.backward()
        optimizer.step()

    avg, best, worst = timer(training_step, repeats=CONV_REPEATS, device=device)

    print(f"\n  Avg time     : {avg:.2f} ms")
    print(f"  Best time    : {best:.2f} ms")
    print(f"  Worst time   : {worst:.2f} ms")
    return avg


# ─────────────────────────────────────────────
# TEST 5: GPU Memory Bandwidth
# ─────────────────────────────────────────────
def run_memory_bandwidth(device):
    print_section("Test 5: GPU Memory Bandwidth")

    size_gb   = 2.0
    n_elements = int(size_gb * 1024**3 / 4)  # float32 = 4 bytes
    repeats    = 20

    print(f"  Tensor size : {size_gb:.1f} GB")
    print(f"  Repeats     : {repeats}  (+ {WARMUP_RUNS} warm-up)")

    A = torch.randn(n_elements, device=device, dtype=torch.float32)
    B = torch.empty_like(A)

    def copy():
        B.copy_(A)

    avg, best, worst = timer(copy, repeats=repeats, warmup=WARMUP_RUNS, device=device)

    # Bandwidth = bytes read + bytes written / time
    bandwidth_gb_s = (2 * size_gb) / (avg / 1000)

    print(f"\n  Avg time     : {avg:.2f} ms")
    print(f"  Bandwidth    : {bandwidth_gb_s:.1f} GB/s")
    return bandwidth_gb_s


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary(results):
    print_header("BENCHMARK SUMMARY")
    print(f"  {'Test':<40} {'Result':>15}")
    print(f"  {'─'*40} {'─'*15}")
    for name, value, unit in results:
        print(f"  {name:<40} {value:>12.2f} {unit}")
    print()
    print("  Save this output and compare it against")
    print("  the same script run inside the NGC container.")
    print("═" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print_header("PyTorch GPU Benchmark  —  gpu_benchmark.py")

    system_info()

    device = torch.device("cuda:0")
    results = []

    # Run all tests
    avg_fp32,  tflops_fp32  = run_matmul_fp32(device)
    avg_fp16,  tflops_fp16  = run_matmul_fp16(device)
    avg_conv                = run_conv2d(device)
    avg_train               = run_training_step(device)
    bandwidth               = run_memory_bandwidth(device)

    # Collect for summary table
    results = [
        ("MatMul FP32 — avg latency",    avg_fp32,    "ms"),
        ("MatMul FP32 — throughput",      tflops_fp32, "TFLOPS"),
        ("MatMul FP16 — avg latency",    avg_fp16,    "ms"),
        ("MatMul FP16 — throughput",      tflops_fp16, "TFLOPS"),
        ("Conv2D — avg latency",          avg_conv,    "ms"),
        ("Training step — avg latency",   avg_train,   "ms"),
        ("Memory bandwidth",              bandwidth,   "GB/s"),
    ]

    print_summary(results)