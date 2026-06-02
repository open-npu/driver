/*
 * Open-NPU Baremetal Driver — High-level Driver API
 * npu_driver.h — Data structures and function declarations
 *
 * Self-contained: does NOT depend on csim headers.
 * Constants and struct layouts are kept consistent with csim/npu_types.h
 * but independently defined for zero-dependency MCU deployment.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_DRIVER_H
#define NPU_DRIVER_H

#include <stdint.h>
#include <stddef.h>
#include "npu_hal.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  NPU1 binary model format constants
 * ═══════════════════════════════════════════════════════════════════ */
#define NPU_MODEL_MAGIC         0x4E505531U  /* "NPU1" */
#define NPU_FIXED_CONFIG_SIZE   62           /* bytes per layer fixed config */

/* Per-channel parameter size: M(2) + S(1) + pad(1) + zp(2) + bias(8) = 14 */
#define NPU_PERCHANNEL_PARAM_SIZE  14

/* Add rescale parameter size: M_A(2) + S_A(1) + pad(1) + M_B(2) + S_B(1) + pad(1) = 8 */
#define NPU_ADD_PARAM_SIZE         8

/* ═══════════════════════════════════════════════════════════════════
 *  Layer descriptor (matches NPU1 binary fixed_config layout, 62 bytes)
 *  Parsed from model binary — all fields are little-endian.
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Offset 0-1: op/data type */
    uint8_t  op_type;        /* 0: OP_xxx */
    uint8_t  data_type;      /* 1: 0=INT8, 1=INT16 */

    /* Offset 2-13: dimensions */
    uint16_t in_h;           /* 2 */
    uint16_t in_w;           /* 4 */
    uint16_t in_c;           /* 6 */
    uint16_t out_h;          /* 8 */
    uint16_t out_w;          /* 10 */
    uint16_t out_c;          /* 12 */

    /* Offset 14-23: convolution */
    uint8_t  kernel_h;       /* 14 */
    uint8_t  kernel_w;       /* 15 */
    uint8_t  dilation_h;     /* 16 */
    uint8_t  dilation_w;     /* 17 */
    uint8_t  stride_h;       /* 18 */
    uint8_t  stride_w;       /* 19 */
    uint8_t  pad_top;        /* 20 */
    uint8_t  pad_bottom;     /* 21 */
    uint8_t  pad_left;       /* 22 */
    uint8_t  pad_right;      /* 23 */

    /* Offset 24-29: pooling */
    uint8_t  pool_mode;      /* 24: 0=Max, 1=Avg */
    uint8_t  pool_h;         /* 25 */
    uint8_t  pool_w;         /* 26 */
    uint8_t  pool_stride_h;  /* 27 */
    uint8_t  pool_stride_w;  /* 28 */
    uint8_t  global_pool;    /* 29 */

    /* Offset 30-34: resize / deconv */
    uint8_t  resize_mode;    /* 30 */
    uint8_t  scale_h;        /* 31: Q4.4 */
    uint8_t  scale_w;        /* 32: Q4.4 */
    uint8_t  insert_h;       /* 33: deconv */
    uint8_t  insert_w;       /* 34: deconv */

    /* Offset 35-38: concat */
    uint16_t concat_offset;  /* 35 */
    uint16_t concat_total_c; /* 37 */

    /* Offset 39-46: tiling */
    uint16_t tile_h;         /* 39 */
    uint16_t tile_w;         /* 41 */
    uint16_t tile_num_h;     /* 43 */
    uint16_t tile_num_w;     /* 45 */

    /* Offset 47-53: post-processing */
    uint8_t  post_ctrl;      /* 47 */
    uint8_t  sched_ctrl;     /* 48 */
    int16_t  clamp_min;      /* 49 */
    int16_t  clamp_max;      /* 51 */
    int8_t   in_zp;          /* 53 */

    /* Offset 54: padding */
    uint8_t  _pad1;          /* 54 */

    /* Offset 55-56: param channel count */
    uint16_t param_ch_count; /* 55 */

    /* Offset 57-61: flags */
    uint8_t  has_lut;        /* 57 */
    uint8_t  has_add;        /* 58 */
    int8_t   residual_src;   /* 59 */
    int16_t  input_src;      /* 60 */
    /* Total: 62 bytes */
} __attribute__((packed)) npu_layer_desc_t;

/* ═══════════════════════════════════════════════════════════════════
 *  NPU1 model header (16 bytes, at offset 0 in binary)
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t magic;          /* 0x4E505531 */
    uint32_t num_layers;
    uint32_t weight_offset;  /* byte offset to weight blob */
    uint32_t weight_size;    /* total weight bytes */
} __attribute__((packed)) npu_model_header_t;

/* ═══════════════════════════════════════════════════════════════════
 *  Hardware info (returned by npu_get_hw_config)
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint8_t  array_size;     /* systolic array N (16/32/64) */
    uint8_t  num_arrays;     /* number of arrays */
    uint8_t  dw_channels;    /* DW conv parallel channels = 2^dw_ch_log2 */
    uint16_t spad_size_kb;   /* scratchpad total KB */
    uint8_t  has_int16;
    uint8_t  has_lut;
    uint8_t  has_ipu;
} npu_hw_config_t;

/* ═══════════════════════════════════════════════════════════════════
 *  Version info
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
} npu_version_t;

/* ═══════════════════════════════════════════════════════════════════
 *  Error codes
 * ═══════════════════════════════════════════════════════════════════ */
typedef enum {
    NPU_OK = 0,
    NPU_ERR_BUSY,
    NPU_ERR_TIMEOUT,
    NPU_ERR_BAD_MODEL,       /* invalid magic or format */
    NPU_ERR_HW_ERROR,        /* hardware error flag set */
} npu_status_t;

/* ═══════════════════════════════════════════════════════════════════
 *  Callback type for IRQ-driven completion
 * ═══════════════════════════════════════════════════════════════════ */
typedef void (*npu_done_callback_t)(uint32_t irq_status, void *user_data);

/* ═══════════════════════════════════════════════════════════════════
 *  API Functions
 * ═══════════════════════════════════════════════════════════════════ */

/* ── Initialization & info ── */
void          npu_init(void);
void          npu_reset(void);
npu_version_t npu_get_version(void);
void          npu_get_hw_config(npu_hw_config_t *cfg);

/* ── Single-layer programming & execution ── */
npu_status_t  npu_program_layer(const npu_layer_desc_t *desc,
                                uint32_t ddr_in_addr,
                                uint32_t ddr_out_addr,
                                uint32_t ddr_wgt_addr,
                                uint32_t ddr_param_addr,
                                uint32_t dma_in_size,
                                uint32_t dma_wgt_size,
                                uint32_t dma_out_size,
                                uint32_t sram_out_base);
npu_status_t  npu_start(void);
npu_status_t  npu_wait_done_poll(uint32_t timeout_cycles);

/* ── IRQ control ── */
void     npu_irq_enable(uint32_t mask);
void     npu_irq_disable(uint32_t mask);
uint32_t npu_irq_status(void);
void     npu_irq_clear(uint32_t mask);
void     npu_set_done_callback(npu_done_callback_t cb, void *user_data);

/* ── IRQ handler (call from ISR) ── */
void     npu_irq_handler(void);

/* ── Performance counters ── */
uint32_t npu_get_cycle_count(void);
uint32_t npu_get_mac_count(void);

/* ── Model execution (NPU1 binary format) ── */
npu_status_t  npu_run_model(const void *model_bin,
                            uint32_t ext_mem_base,
                            uint32_t io_buf_base);

#ifdef __cplusplus
}
#endif

#endif /* NPU_DRIVER_H */
