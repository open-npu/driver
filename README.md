# Open-NPU Driver

Baremetal driver SDK for Open-NPU hardware accelerator.

## Architecture

```
driver/
└── baremetal/
    ├── npu_hal.h      — CSR register addresses, bit fields, MMIO read/write
    ├── npu_hal.c      — HAL compilation unit
    ├── npu_driver.h   — Data structures (npu_layer_desc_t) and API declarations
    └── npu_driver.c   — Full driver implementation
```

Two-layer design:

- **HAL layer** (`npu_hal.h`): Register address macros (`NPU_REG_*`), bit field definitions, packing macros (`LAYER_MODE()`, `DIM_HW()`, `KERNEL_PACK()`, etc.), inline MMIO access via `volatile uint32_t*`.
- **Driver layer** (`npu_driver.h/c`): High-level API for initialization, layer programming, execution control, IRQ management, and NPU1 model binary execution.

## Key Design Principles

- **Zero dependency** — Does not `#include` any csim or OS headers. All constants are independently defined for MCU toolchain portability.
- **Packed struct mapping** — `npu_layer_desc_t` is a 62-byte `__attribute__((packed))` struct that directly maps the NPU1 binary fixed_config layout. Model binary can be cast in-place (zero-copy from Flash).
- **CSR sequence parity** — `npu_program_layer()` writes CSR registers in the same order as the verified RTL testbench (`rtl/tb/test_npu_dma_e2e.py::program_generic_layer()`).

## API Overview

### Initialization

```c
npu_init();                        // Software reset + clear IRQs

npu_version_t ver = npu_get_version();
npu_hw_config_t cfg;
npu_get_hw_config(&cfg);           // Read array_size, spad_size_kb, etc.
```

### Single-layer Execution (Polling)

```c
npu_layer_desc_t desc = { ... };   // Fill or cast from NPU1 binary

npu_program_layer(&desc,
    ddr_in, ddr_out, ddr_wgt, ddr_param,
    in_size, wgt_size, out_size,
    sram_out_base);

npu_start();
npu_status_t st = npu_wait_done_poll(1000000);
```

### IRQ-driven Execution

```c
void my_callback(uint32_t irq_status, void *data) {
    // Handle completion
}

npu_set_done_callback(my_callback, NULL);
npu_irq_enable(IRQ_DONE_EN);
npu_start();
// ... CPU sleeps or does other work ...

// In your ISR vector:
void npu_isr(void) {
    npu_irq_handler();  // Clears IRQ + invokes callback
}
```

### Run NPU1 Model Binary

```c
// model_bin: pointer to NPU1 binary (Flash or RAM)
// io_buf_base: DDR buffer for input/output activations
npu_status_t st = npu_run_model(model_bin, ext_mem_base, io_buf_base);
```

## Supported Operations

All 8 operator types supported in `npu_program_layer()`:

| op_type | Operator | Op-specific Register |
|---------|----------|---------------------|
| 0 | Conv2D | — |
| 1 | DWConv | — |
| 2 | FC | — |
| 3 | Pooling (Max/Avg/Global) | `NPU_REG_POOL_CFG` |
| 4 | Eltwise Add | `NPU_REG_DMA_ADD_B_ADDR` |
| 5 | Resize (nearest/bilinear) | `NPU_REG_RESIZE_CFG` |
| 6 | Deconv | `NPU_REG_DECONV_CFG` |
| 7 | Concat | `NPU_REG_CONCAT_CFG` |

## Build

```bash
# Cross-compile for RISC-V (example)
riscv32-unknown-elf-gcc -Wall -Wextra -std=c11 -ffreestanding \
    -DNPU_BASE_ADDR=0x80000000 \
    -c baremetal/npu_driver.c -o npu_driver.o

# Host compile check
gcc -Wall -Wextra -Wpedantic -std=c11 -c baremetal/npu_driver.c -o /dev/null
```

Override `NPU_BASE_ADDR` via `-D` flag to match your SoC memory map.

## License

Apache-2.0
