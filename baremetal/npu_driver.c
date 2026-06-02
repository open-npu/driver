/*
 * Open-NPU Baremetal Driver — Implementation
 * npu_driver.c
 *
 * Self-contained: does NOT depend on csim headers.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_driver.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Static state for IRQ callback
 * ═══════════════════════════════════════════════════════════════════ */
static npu_done_callback_t s_done_cb   = NULL;
static void               *s_done_data = NULL;

/* ═══════════════════════════════════════════════════════════════════
 *  Initialization & info
 * ═══════════════════════════════════════════════════════════════════ */

void npu_init(void)
{
    npu_reset();
}

void npu_reset(void)
{
    /* Software reset — auto-clears */
    npu_reg_write(NPU_REG_CTRL, CTRL_SOFT_RST);

    /* Clear any pending IRQs */
    npu_reg_write(NPU_REG_IRQ_STATUS,
                  IRQ_DONE_EN | IRQ_ERROR_EN | IRQ_DMA_DONE_EN);

    /* Disable all IRQs */
    npu_reg_write(NPU_REG_IRQ_EN, 0);

    s_done_cb   = NULL;
    s_done_data = NULL;
}

npu_version_t npu_get_version(void)
{
    uint32_t v = npu_reg_read(NPU_REG_VERSION);
    npu_version_t ver;
    ver.major = VER_MAJOR(v);
    ver.minor = VER_MINOR(v);
    ver.patch = VER_PATCH(v);
    return ver;
}

void npu_get_hw_config(npu_hw_config_t *cfg)
{
    uint32_t v = npu_reg_read(NPU_REG_HW_CONFIG);
    cfg->array_size  = (uint8_t)HW_CFG_ARRAY_SIZE(v);
    cfg->num_arrays  = (uint8_t)HW_CFG_NUM_ARRAYS(v);
    cfg->dw_channels = (uint8_t)(1U << HW_CFG_DW_CH_LOG2(v));
    cfg->spad_size_kb = (uint16_t)(HW_CFG_SPAD_4KB(v) * 4);
    cfg->has_int16   = (uint8_t)HW_CFG_HAS_INT16(v);
    cfg->has_lut     = (uint8_t)HW_CFG_HAS_LUT(v);
    cfg->has_ipu     = (uint8_t)HW_CFG_HAS_IPU(v);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Single-layer programming & execution
 * ═══════════════════════════════════════════════════════════════════ */

npu_status_t npu_program_layer(const npu_layer_desc_t *d,
                               uint32_t ddr_in_addr,
                               uint32_t ddr_out_addr,
                               uint32_t ddr_wgt_addr,
                               uint32_t ddr_param_addr,
                               uint32_t dma_in_size,
                               uint32_t dma_wgt_size,
                               uint32_t dma_out_size,
                               uint32_t sram_out_base)
{
    /* Check NPU is not busy */
    if (npu_reg_read(NPU_REG_STATUS) & STATUS_BUSY)
        return NPU_ERR_BUSY;

    /* ── Group 1: Layer parameters ── */

    npu_reg_write(NPU_REG_LAYER_MODE,
                  LAYER_MODE(d->op_type, d->data_type));

    npu_reg_write(NPU_REG_IN_DIM_HW,  DIM_HW(d->in_h, d->in_w));
    npu_reg_write(NPU_REG_IN_DIM_C,   d->in_c);
    npu_reg_write(NPU_REG_OUT_DIM_HW, DIM_HW(d->out_h, d->out_w));
    npu_reg_write(NPU_REG_OUT_DIM_C,  d->out_c);

    npu_reg_write(NPU_REG_KERNEL_SIZE,
                  KERNEL_PACK(d->kernel_h, d->kernel_w,
                              d->dilation_h, d->dilation_w));

    npu_reg_write(NPU_REG_STRIDE,
                  STRIDE_PACK(d->stride_h, d->stride_w));

    npu_reg_write(NPU_REG_PADDING,
                  PADDING_PACK(d->pad_top, d->pad_bottom,
                               d->pad_left, d->pad_right));

    /* ── Op-specific registers ── */
    switch (d->op_type) {
    case NPU_OP_POOLING:
        npu_reg_write(NPU_REG_POOL_CFG,
                      POOL_CFG_PACK(d->pool_mode, d->pool_h, d->pool_w,
                                    d->pool_stride_h, d->pool_stride_w,
                                    d->global_pool));
        break;
    case NPU_OP_RESIZE:
        npu_reg_write(NPU_REG_RESIZE_CFG,
                      RESIZE_CFG_PACK(d->resize_mode,
                                      d->scale_h, d->scale_w));
        break;
    case NPU_OP_DECONV:
        npu_reg_write(NPU_REG_DECONV_CFG,
                      DECONV_CFG_PACK(d->insert_h, d->insert_w));
        break;
    case NPU_OP_CONCAT:
        npu_reg_write(NPU_REG_CONCAT_CFG,
                      CONCAT_CFG_PACK(d->concat_offset, d->concat_total_c));
        break;
    default:
        break;
    }

    /* ── Tiling ── */
    npu_reg_write(NPU_REG_TILE_CFG,
                  TILE_CFG_PACK(d->tile_h, d->tile_w));
    npu_reg_write(NPU_REG_TILE_COUNT,
                  TILE_COUNT_PACK(d->tile_num_h, d->tile_num_w));

    /* ── SRAM base ── */
    npu_reg_write(NPU_REG_SRAM_BASE,
                  SRAM_BASE_PACK(0, sram_out_base));

    /* ── Group 2: DMA configuration ── */
    npu_reg_write(NPU_REG_DMA_IN_ADDR,   ddr_in_addr);
    npu_reg_write(NPU_REG_DMA_OUT_ADDR,  ddr_out_addr);
    npu_reg_write(NPU_REG_DMA_WGT_ADDR,  ddr_wgt_addr);
    npu_reg_write(NPU_REG_DMA_PARAM_ADDR, ddr_param_addr);

    npu_reg_write(NPU_REG_DMA_IN_SIZE,   dma_in_size);
    npu_reg_write(NPU_REG_DMA_WGT_SIZE,  dma_wgt_size);
    npu_reg_write(NPU_REG_DMA_OUT_SIZE,  dma_out_size);

    npu_reg_write(NPU_REG_DMA_CTRL, 0);  /* no transpose, default burst */

    /* ── Group 3: Post-processing ── */
    npu_reg_write(NPU_REG_POST_CTRL, d->post_ctrl);
    npu_reg_write(NPU_REG_POST_PARAM_COUNT, d->param_ch_count);
    npu_reg_write(NPU_REG_POST_CLAMP,
                  POST_CLAMP_PACK(d->clamp_min, d->clamp_max));

    return NPU_OK;
}

npu_status_t npu_start(void)
{
    if (npu_reg_read(NPU_REG_STATUS) & STATUS_BUSY)
        return NPU_ERR_BUSY;
    npu_reg_write(NPU_REG_CTRL, CTRL_START);
    return NPU_OK;
}

npu_status_t npu_wait_done_poll(uint32_t timeout_cycles)
{
    for (uint32_t i = 0; i < timeout_cycles; i++) {
        uint32_t st = npu_reg_read(NPU_REG_STATUS);
        if (st & STATUS_ERROR)
            return NPU_ERR_HW_ERROR;
        if (st & STATUS_DONE)
            return NPU_OK;
    }
    return NPU_ERR_TIMEOUT;
}

/* ═══════════════════════════════════════════════════════════════════
 *  IRQ control
 * ═══════════════════════════════════════════════════════════════════ */

void npu_irq_enable(uint32_t mask)
{
    uint32_t cur = npu_reg_read(NPU_REG_IRQ_EN);
    npu_reg_write(NPU_REG_IRQ_EN, cur | mask);
}

void npu_irq_disable(uint32_t mask)
{
    uint32_t cur = npu_reg_read(NPU_REG_IRQ_EN);
    npu_reg_write(NPU_REG_IRQ_EN, cur & ~mask);
}

uint32_t npu_irq_status(void)
{
    return npu_reg_read(NPU_REG_IRQ_STATUS);
}

void npu_irq_clear(uint32_t mask)
{
    npu_reg_write(NPU_REG_IRQ_STATUS, mask);
}

void npu_set_done_callback(npu_done_callback_t cb, void *user_data)
{
    s_done_cb   = cb;
    s_done_data = user_data;
}

void npu_irq_handler(void)
{
    uint32_t st = npu_reg_read(NPU_REG_IRQ_STATUS);

    /* Clear all pending IRQs (W1C) */
    npu_reg_write(NPU_REG_IRQ_STATUS, st);

    /* Invoke user callback if registered */
    if (s_done_cb != NULL)
        s_done_cb(st, s_done_data);
}

/* ═══════════════════════════════════════════════════════════════════
 *  Performance counters
 * ═══════════════════════════════════════════════════════════════════ */

uint32_t npu_get_cycle_count(void)
{
    return npu_reg_read(NPU_REG_PERF_CNT);
}

uint32_t npu_get_mac_count(void)
{
    return npu_reg_read(NPU_REG_MAC_CNT);
}

/* ═══════════════════════════════════════════════════════════════════
 *  NPU1 model binary parsing helpers
 * ═══════════════════════════════════════════════════════════════════ */

/* Little-endian byte access helpers */
static inline uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static inline uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/*
 * Compute total variable-length data after the 62-byte fixed_config.
 */
static uint32_t layer_var_size(const npu_layer_desc_t *d)
{
    uint32_t size = 0;
    size += (uint32_t)d->param_ch_count * NPU_PERCHANNEL_PARAM_SIZE;
    if (d->has_add)
        size += NPU_ADD_PARAM_SIZE;
    if (d->has_lut)
        size += 256 + 512;  /* lut_i8(256) + lut_i16(512) */
    return size;
}

/*
 * Compute weight size for a given layer (depends on op type).
 */
static uint32_t layer_weight_size(const npu_layer_desc_t *d)
{
    uint32_t bpe = (d->data_type == NPU_DTYPE_INT16) ? 2 : 1;
    switch (d->op_type) {
    case NPU_OP_CONV2D:
    case NPU_OP_FC:
    case NPU_OP_DECONV:
        return (uint32_t)d->out_c * d->in_c * d->kernel_h * d->kernel_w * bpe;
    case NPU_OP_DW_CONV:
        return (uint32_t)d->in_c * d->kernel_h * d->kernel_w * bpe;
    default:
        return 0;  /* Pooling, Add, Resize, Concat have no weights */
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  Model execution (NPU1 binary format)
 * ═══════════════════════════════════════════════════════════════════ */

npu_status_t npu_run_model(const void *model_bin,
                           uint32_t ext_mem_base,
                           uint32_t io_buf_base)
{
    (void)ext_mem_base;  /* reserved for future use (e.g., absolute address remapping) */
    const uint8_t *base = (const uint8_t *)model_bin;

    /* Validate header */
    const npu_model_header_t *hdr = (const npu_model_header_t *)base;
    if (hdr->magic != NPU_MODEL_MAGIC)
        return NPU_ERR_BAD_MODEL;

    uint32_t num_layers  = hdr->num_layers;
    uint32_t weight_off  = hdr->weight_offset;

    /* Descriptor cursor (right after 16-byte header) */
    const uint8_t *desc_ptr = base + 16;

    /* Weight cursor (within model binary, memory-mapped) */
    const uint8_t *wgt_ptr = base + weight_off;

    /*
     * I/O buffer management:
     *   - Ping-pong between two halves of io_buf_base.
     *   - Layer 0 reads from io_buf_base (user pre-loads input there).
     *   - Each subsequent layer reads from previous layer's output.
     *   - Final layer output ends up at one of the two buffers.
     */
    uint32_t buf_a = io_buf_base;
    uint32_t buf_b = io_buf_base + 512 * 1024;  /* 512 KB stride */
    uint32_t cur_in  = buf_a;
    uint32_t cur_out;
    int use_a = 0;  /* first output goes to buf_b */

    for (uint32_t i = 0; i < num_layers; i++) {
        /* The fixed config is directly castable (packed struct) */
        const npu_layer_desc_t *d = (const npu_layer_desc_t *)desc_ptr;

        /* Compute sizes */
        uint32_t bpe = (d->data_type == NPU_DTYPE_INT16) ? 2 : 1;
        uint32_t in_size  = (uint32_t)d->in_h  * d->in_w  * d->in_c  * bpe;
        uint32_t out_size = (uint32_t)d->out_h * d->out_w * d->out_c * bpe;
        uint32_t wgt_size = layer_weight_size(d);

        /* Output buffer */
        cur_out = use_a ? buf_a : buf_b;

        /* Param address (per-channel params follow fixed config in binary) */
        uint32_t param_addr = 0;
        if (d->param_ch_count > 0)
            param_addr = (uint32_t)(uintptr_t)(desc_ptr + NPU_FIXED_CONFIG_SIZE);

        /* SRAM out_base: simple heuristic — place output after input */
        uint32_t n_in_words = in_size / 4;
        if (in_size % 4) n_in_words++;

        /* Program CSR registers */
        npu_status_t st;
        st = npu_program_layer(d,
                               cur_in,              /* ddr_in_addr */
                               cur_out,             /* ddr_out_addr */
                               (uint32_t)(uintptr_t)wgt_ptr, /* ddr_wgt_addr */
                               param_addr,          /* ddr_param_addr */
                               in_size,             /* dma_in_size */
                               wgt_size,            /* dma_wgt_size */
                               out_size,            /* dma_out_size */
                               n_in_words);         /* sram_out_base */
        if (st != NPU_OK)
            return st;

        /* Add-specific: program second input address */
        if (d->op_type == NPU_OP_ELTWISE_ADD && d->residual_src >= 0) {
            /* For simple sequential models, residual is not handled.
             * In production, track layer output addresses for residual routing. */
            npu_reg_write(NPU_REG_DMA_ADD_B_ADDR, cur_in);
        }

        /* Start and wait */
        st = npu_start();
        if (st != NPU_OK)
            return st;

        st = npu_wait_done_poll(0xFFFFFFFFU);
        if (st != NPU_OK)
            return st;

        /* Advance pointers */
        cur_in = cur_out;
        use_a = !use_a;

        /* Advance weight cursor */
        wgt_ptr += wgt_size;

        /* Advance descriptor cursor past fixed + variable data */
        uint32_t var_size = layer_var_size(d);
        desc_ptr += NPU_FIXED_CONFIG_SIZE + var_size;
    }

    return NPU_OK;
}
