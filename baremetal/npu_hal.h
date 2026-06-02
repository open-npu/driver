/*
 * Open-NPU Baremetal Driver — Hardware Abstraction Layer
 * npu_hal.h — CSR register definitions and MMIO access
 *
 * Self-contained: does NOT depend on csim headers.
 * Register addresses match npu-register-spec.md V1.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_HAL_H
#define NPU_HAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  Base address (override via -DNPU_BASE_ADDR=0x...)
 * ═══════════════════════════════════════════════════════════════════ */
#ifndef NPU_BASE_ADDR
#define NPU_BASE_ADDR  0x80000000UL
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  MMIO read/write primitives
 * ═══════════════════════════════════════════════════════════════════ */
static inline void npu_reg_write(uint32_t offset, uint32_t value) {
    *(volatile uint32_t *)(NPU_BASE_ADDR + offset) = value;
}

static inline uint32_t npu_reg_read(uint32_t offset) {
    return *(volatile uint32_t *)(NPU_BASE_ADDR + offset);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Group 0: Control & Status (0x000 - 0x03F)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG_CTRL            0x000
#define NPU_REG_STATUS          0x004
#define NPU_REG_IRQ_EN          0x008
#define NPU_REG_IRQ_STATUS      0x00C
#define NPU_REG_ERROR_CODE      0x010
#define NPU_REG_VERSION         0x014
#define NPU_REG_HW_CONFIG       0x018
#define NPU_REG_PERF_CNT        0x01C
#define NPU_REG_MAC_CNT         0x020
#define NPU_REG_SERIAL_LO       0x024
#define NPU_REG_SERIAL_HI       0x028
#define NPU_REG_VENDOR_ID       0x02C
#define NPU_REG_LAYER_COUNT     0x030

/* CTRL bit definitions */
#define CTRL_START      (1U << 0)
#define CTRL_ABORT      (1U << 1)
#define CTRL_SOFT_RST   (1U << 2)
#define CTRL_AUTO_NEXT  (1U << 3)

/* STATUS bit definitions */
#define STATUS_BUSY         (1U << 0)
#define STATUS_DMA_BUSY     (1U << 1)
#define STATUS_ERROR        (1U << 2)
#define STATUS_DONE         (1U << 3)
#define STATUS_CURR_LAYER_SHIFT  8
#define STATUS_CURR_LAYER_MASK   0xFF

/* IRQ_EN / IRQ_STATUS bit definitions */
#define IRQ_DONE_EN     (1U << 0)
#define IRQ_ERROR_EN    (1U << 1)
#define IRQ_DMA_DONE_EN (1U << 2)

/* HW_CONFIG field extraction helpers */
#define HW_CFG_ARRAY_SIZE(v)   ((v) & 0xFF)
#define HW_CFG_NUM_ARRAYS(v)   (((v) >> 8) & 0x0F)
#define HW_CFG_DW_CH_LOG2(v)   (((v) >> 12) & 0x0F)
#define HW_CFG_SPAD_4KB(v)     (((v) >> 16) & 0xFF)
#define HW_CFG_HAS_INT16(v)    (((v) >> 24) & 1)
#define HW_CFG_HAS_LUT(v)      (((v) >> 25) & 1)
#define HW_CFG_HAS_IPU(v)      (((v) >> 26) & 1)

/* VERSION field extraction */
#define VER_MAJOR(v)   (((v) >> 16) & 0xFF)
#define VER_MINOR(v)   (((v) >> 8) & 0xFF)
#define VER_PATCH(v)   ((v) & 0xFF)

/* ═══════════════════════════════════════════════════════════════════
 *  Group 1: Layer Configuration (0x040 - 0x0FF)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG_LAYER_MODE      0x040
#define NPU_REG_IN_DIM_HW       0x044
#define NPU_REG_IN_DIM_C        0x048
#define NPU_REG_OUT_DIM_HW      0x04C
#define NPU_REG_OUT_DIM_C       0x050
#define NPU_REG_KERNEL_SIZE     0x054
#define NPU_REG_STRIDE          0x058
#define NPU_REG_PADDING         0x05C
#define NPU_REG_POOL_CFG        0x060
#define NPU_REG_RESIZE_CFG      0x064
#define NPU_REG_DECONV_CFG      0x068
#define NPU_REG_CONCAT_CFG      0x06C
#define NPU_REG_TILE_CFG        0x070
#define NPU_REG_TILE_COUNT      0x074
#define NPU_REG_SRAM_BASE       0x078

/* LAYER_MODE field packing */
#define LAYER_MODE(op, dtype)  ((uint32_t)(op) | ((uint32_t)(dtype) << 4))

/* Dimension packing (H in [15:0], W in [31:16]) */
#define DIM_HW(h, w)   (((uint32_t)(h) & 0xFFFF) | (((uint32_t)(w) & 0xFFFF) << 16))

/* Kernel packing: [7:0]=KH, [15:8]=KW, [23:16]=DH, [31:24]=DW */
#define KERNEL_PACK(kh, kw, dh, dw)  \
    ((uint32_t)(kh) | ((uint32_t)(kw) << 8) | \
     ((uint32_t)(dh) << 16) | ((uint32_t)(dw) << 24))

/* Stride packing: [7:0]=SH, [15:8]=SW */
#define STRIDE_PACK(sh, sw)  ((uint32_t)(sh) | ((uint32_t)(sw) << 8))

/* Padding packing: [7:0]=top, [15:8]=bottom, [23:16]=left, [31:24]=right */
#define PADDING_PACK(top, bot, left, right)  \
    ((uint32_t)(top) | ((uint32_t)(bot) << 8) | \
     ((uint32_t)(left) << 16) | ((uint32_t)(right) << 24))

/* SRAM_BASE packing: [12:0]=act_base, [28:16]=out_base */
#define SRAM_BASE_PACK(act, out)  \
    (((uint32_t)(act) & 0x1FFF) | (((uint32_t)(out) & 0x1FFF) << 16))

/* Tile config packing */
#define TILE_CFG_PACK(th, tw)     (((uint32_t)(th) & 0xFFFF) | (((uint32_t)(tw) & 0xFFFF) << 16))
#define TILE_COUNT_PACK(nh, nw)   (((uint32_t)(nh) & 0xFFFF) | (((uint32_t)(nw) & 0xFFFF) << 16))

/* Pool config packing */
#define POOL_CFG_PACK(mode, ph, pw, sh, sw, global)  \
    ((uint32_t)(mode) | ((uint32_t)(ph) << 4) | ((uint32_t)(pw) << 8) | \
     ((uint32_t)(sh) << 12) | ((uint32_t)(sw) << 16) | ((uint32_t)(global) << 20))

/* Resize config packing */
#define RESIZE_CFG_PACK(mode, scale_h, scale_w)  \
    ((uint32_t)(mode) | ((uint32_t)(scale_h) << 8) | ((uint32_t)(scale_w) << 16))

/* Deconv config packing */
#define DECONV_CFG_PACK(insert_h, insert_w)  \
    ((uint32_t)(insert_h) | ((uint32_t)(insert_w) << 8))

/* Concat config packing */
#define CONCAT_CFG_PACK(offset, total_c)  \
    (((uint32_t)(offset) & 0xFFFF) | (((uint32_t)(total_c) & 0xFFFF) << 16))

/* ═══════════════════════════════════════════════════════════════════
 *  Group 2: DMA Configuration (0x100 - 0x17F)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG_DMA_IN_ADDR         0x100
#define NPU_REG_DMA_OUT_ADDR        0x104
#define NPU_REG_DMA_WGT_ADDR        0x108
#define NPU_REG_DMA_PARAM_ADDR      0x10C
#define NPU_REG_DMA_IN_STRIDE       0x110
#define NPU_REG_DMA_OUT_STRIDE      0x114
#define NPU_REG_DMA_CTRL            0x118
#define NPU_REG_DMA_STATUS          0x11C
#define NPU_REG_DMA_ADD_B_ADDR      0x120
#define NPU_REG_DMA_ADD_PARAM_ADDR  0x124
#define NPU_REG_DMA_IN_SIZE         0x128
#define NPU_REG_DMA_WGT_SIZE        0x12C
#define NPU_REG_DMA_OUT_SIZE        0x130

/* ═══════════════════════════════════════════════════════════════════
 *  Group 3: Post-processing (0x180 - 0x1FF)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG_POST_CTRL           0x180
#define NPU_REG_POST_PARAM_ADDR     0x184
#define NPU_REG_POST_PARAM_COUNT    0x188
#define NPU_REG_POST_CLAMP          0x18C
#define NPU_REG_POST_ACT_CFG        0x190
#define NPU_REG_POST_ADD_PARAM_ADDR 0x194
#define NPU_REG_POST_ADD_INPUT_ADDR 0x198
#define NPU_REG_POST_ADD_STRIDE     0x19C

/* POST_CTRL bit definitions */
#define POST_PPU_MODE_MASK   0x03U
#define PPU_MODE_CONV_REQ    0U
#define PPU_MODE_ADD         1U
#define PPU_MODE_RELU_ONLY   2U
#define PPU_MODE_PASSTHROUGH 3U
#define POST_RELU_EN         (1U << 2)
#define POST_RELU6_EN        (1U << 3)
#define POST_LUT_EN          (1U << 4)
#define POST_ZP_EN           (1U << 5)
#define POST_BIAS_EN         (1U << 6)
#define POST_INT16_OUT       (1U << 7)

/* POST_CLAMP packing: [15:0]=min (signed), [31:16]=max (signed) */
#define POST_CLAMP_PACK(min_val, max_val)  \
    (((uint32_t)(uint16_t)(min_val)) | (((uint32_t)(uint16_t)(max_val)) << 16))

/* ═══════════════════════════════════════════════════════════════════
 *  Group 4: LUT Data (0x200 - 0x3FF)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_REG_LUT_BASE    0x200
#define NPU_LUT_ENTRIES     256

/* ═══════════════════════════════════════════════════════════════════
 *  Operator types (match CSR LAYER_MODE.OP_TYPE)
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_OP_CONV2D       0
#define NPU_OP_DW_CONV      1
#define NPU_OP_FC           2
#define NPU_OP_POOLING      3
#define NPU_OP_ELTWISE_ADD  4
#define NPU_OP_RESIZE       5
#define NPU_OP_DECONV       6
#define NPU_OP_CONCAT       7

/* Data types */
#define NPU_DTYPE_INT8      0
#define NPU_DTYPE_INT16     1

#ifdef __cplusplus
}
#endif

#endif /* NPU_HAL_H */
