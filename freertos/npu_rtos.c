/*
 * Open-NPU FreeRTOS Driver — Implementation
 * npu_rtos.c — Mutex-protected, IRQ-driven NPU driver for FreeRTOS
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "npu_rtos.h"

/* ═══════════════════════════════════════════════════════════════════
 *  Internal state
 * ═══════════════════════════════════════════════════════════════════ */
static SemaphoreHandle_t  s_hw_mutex;       /* hardware exclusive access   */
static TaskHandle_t       s_caller_task;    /* task waiting for completion */
static volatile uint32_t  s_irq_result;     /* IRQ status from ISR        */
static volatile uint8_t   s_initialized;
static uint32_t           s_default_timeout_ms;

/* ═══════════════════════════════════════════════════════════════════
 *  ISR callback — registered with baremetal npu_set_done_callback()
 *
 *  Called from npu_irq_handler() in ISR context.
 *  Wakes the waiting task via direct-to-task notification.
 * ═══════════════════════════════════════════════════════════════════ */
static void rtos_irq_callback(uint32_t irq_status, void *user_data)
{
    (void)user_data;
    BaseType_t woken = pdFALSE;

    s_irq_result = irq_status;

    if (s_caller_task != NULL) {
        xTaskNotifyFromISR(s_caller_task, irq_status,
                           eSetValueWithOverwrite, &woken);
        portYIELD_FROM_ISR(woken);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 *  Helpers — model parsing (shared between sync & async)
 *
 *  These mirror the baremetal npu_run_model() logic but replace
 *  polling wait with xTaskNotifyWait().
 * ═══════════════════════════════════════════════════════════════════ */

static uint32_t rtos_layer_var_size(const npu_layer_desc_t *d)
{
    uint32_t size = 0;
    size += (uint32_t)d->param_ch_count * NPU_PERCHANNEL_PARAM_SIZE;
    if (d->has_add)
        size += NPU_ADD_PARAM_SIZE;
    if (d->has_lut)
        size += 256 + 512;  /* lut_i8(256) + lut_i16(512) */
    return size;
}

static uint32_t rtos_layer_weight_size(const npu_layer_desc_t *d)
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
        return 0;
    }
}

/*
 * Run all layers of a model, using xTaskNotifyWait for completion.
 * Caller must already hold s_hw_mutex and have set s_caller_task.
 */
static npu_status_t rtos_run_layers(const void *model_bin,
                                    uint32_t io_buf_base,
                                    uint32_t timeout_ms)
{
    const uint8_t *base = (const uint8_t *)model_bin;
    const npu_model_header_t *hdr = (const npu_model_header_t *)base;

    if (hdr->magic != NPU_MODEL_MAGIC)
        return NPU_ERR_BAD_MODEL;

    uint32_t num_layers = hdr->num_layers;
    uint32_t weight_off = hdr->weight_offset;

    const uint8_t *desc_ptr = base + 16;
    const uint8_t *wgt_ptr  = base + weight_off;

    /* Ping-pong I/O buffers */
    uint32_t buf_a  = io_buf_base;
    uint32_t buf_b  = io_buf_base + 512 * 1024;
    uint32_t cur_in = buf_a;
    uint32_t cur_out;
    int use_a = 0;

    TickType_t ticks = (timeout_ms > 0) ? pdMS_TO_TICKS(timeout_ms)
                                        : portMAX_DELAY;

    for (uint32_t i = 0; i < num_layers; i++) {
        const npu_layer_desc_t *d = (const npu_layer_desc_t *)desc_ptr;

        uint32_t bpe      = (d->data_type == NPU_DTYPE_INT16) ? 2 : 1;
        uint32_t in_size  = (uint32_t)d->in_h  * d->in_w  * d->in_c  * bpe;
        uint32_t out_size = (uint32_t)d->out_h * d->out_w * d->out_c * bpe;
        uint32_t wgt_size = rtos_layer_weight_size(d);

        cur_out = use_a ? buf_a : buf_b;

        /* Param address */
        uint32_t param_addr = 0;
        if (d->param_ch_count > 0)
            param_addr = (uint32_t)(uintptr_t)(desc_ptr + NPU_FIXED_CONFIG_SIZE);

        /* SRAM out_base */
        uint32_t n_in_words = in_size / 4;
        if (in_size % 4) n_in_words++;

        /* Clear any stale notification before starting this layer */
        xTaskNotifyWait(0xFFFFFFFF, 0xFFFFFFFF, NULL, 0);

        /* Program layer */
        npu_status_t st;
        st = npu_program_layer(d,
                               cur_in, cur_out,
                               (uint32_t)(uintptr_t)wgt_ptr,
                               param_addr,
                               in_size, wgt_size, out_size,
                               n_in_words);
        if (st != NPU_OK)
            return st;

        /* Add-specific second input */
        if (d->op_type == NPU_OP_ELTWISE_ADD && d->residual_src >= 0)
            npu_reg_write(NPU_REG_DMA_ADD_B_ADDR, cur_in);

        /* Start HW */
        st = npu_start();
        if (st != NPU_OK)
            return st;

        /* Wait for IRQ-driven completion (CPU yields here) */
        uint32_t notif_value = 0;
        BaseType_t got = xTaskNotifyWait(0, 0xFFFFFFFF, &notif_value, ticks);

        if (got != pdPASS)
            return NPU_ERR_TIMEOUT;

        /* Check for hardware error in IRQ status */
        if (notif_value & IRQ_ERROR_EN)
            return NPU_ERR_HW_ERROR;

        /* Advance pointers */
        cur_in = cur_out;
        use_a  = !use_a;
        wgt_ptr  += wgt_size;
        desc_ptr += NPU_FIXED_CONFIG_SIZE + rtos_layer_var_size(d);
    }

    return NPU_OK;
}

/* ═══════════════════════════════════════════════════════════════════
 *  Public API
 * ═══════════════════════════════════════════════════════════════════ */

npu_status_t npu_rtos_init(const npu_rtos_config_t *config)
{
    if (s_initialized)
        return NPU_ERR_BUSY;

    /* Initialize baremetal layer */
    npu_init();

    /* Create hardware mutex */
    s_hw_mutex = xSemaphoreCreateMutex();
    if (s_hw_mutex == NULL)
        return NPU_ERR_HW_ERROR;  /* allocation failure */

    /* Register ISR callback */
    s_caller_task = NULL;
    s_irq_result  = 0;
    npu_set_done_callback(rtos_irq_callback, NULL);

    /* Enable IRQs */
    npu_irq_enable(IRQ_DONE_EN | IRQ_ERROR_EN);

    /* Config */
    s_default_timeout_ms = config ? config->default_timeout_ms : 0;

    s_initialized = 1;
    return NPU_OK;
}

void npu_rtos_deinit(void)
{
    if (!s_initialized)
        return;

    /* Disable IRQs and clear callback */
    npu_irq_disable(IRQ_DONE_EN | IRQ_ERROR_EN);
    npu_set_done_callback(NULL, NULL);

    /* Delete mutex */
    if (s_hw_mutex != NULL) {
        vSemaphoreDelete(s_hw_mutex);
        s_hw_mutex = NULL;
    }

    s_caller_task = NULL;
    s_initialized = 0;
}

npu_status_t npu_rtos_run_model(const void *model_bin,
                                uint32_t ext_mem_base,
                                uint32_t io_buf_base,
                                uint32_t timeout_ms)
{
    (void)ext_mem_base;

    if (!s_initialized)
        return NPU_ERR_BUSY;

    /* Acquire hardware lock (blocks if another task holds it) */
    if (xSemaphoreTake(s_hw_mutex, portMAX_DELAY) != pdPASS)
        return NPU_ERR_BUSY;

    /* Record caller for ISR notification */
    s_caller_task = xTaskGetCurrentTaskHandle();

    /* Use specified timeout, or fall back to default */
    uint32_t tmo = timeout_ms > 0 ? timeout_ms : s_default_timeout_ms;

    /* Run all layers */
    npu_status_t result = rtos_run_layers(model_bin, io_buf_base, tmo);

    /* Cleanup */
    s_caller_task = NULL;
    xSemaphoreGive(s_hw_mutex);

    return result;
}

npu_status_t npu_rtos_run_model_async(npu_inference_req_t *req)
{
    if (!s_initialized || req == NULL)
        return NPU_ERR_BUSY;

    /* Acquire hardware lock */
    if (xSemaphoreTake(s_hw_mutex, portMAX_DELAY) != pdPASS)
        return NPU_ERR_BUSY;

    /* Set up request state */
    req->_caller_task = xTaskGetCurrentTaskHandle();
    req->_submitted   = 1;
    req->_done        = 0;
    req->result       = NPU_OK;
    req->cycles       = 0;

    /* Register caller for notification */
    s_caller_task = req->_caller_task;

    /* Validate model header before starting */
    const npu_model_header_t *hdr = (const npu_model_header_t *)req->model_bin;
    if (hdr->magic != NPU_MODEL_MAGIC) {
        s_caller_task = NULL;
        xSemaphoreGive(s_hw_mutex);
        req->result = NPU_ERR_BAD_MODEL;
        req->_done  = 1;
        return NPU_ERR_BAD_MODEL;
    }

    /*
     * For async, we start the full model execution in the current task context.
     * The actual async behavior comes from the caller being free to do other
     * work between submission and npu_rtos_wait(). Since the NPU is layer-
     * sequential, we run layers inside rtos_wait() rather than here.
     *
     * This design keeps the mutex held during the entire inference to prevent
     * concurrent hardware access. The calling task yields CPU during each
     * layer's xTaskNotifyWait().
     */

    return NPU_OK;
}

npu_status_t npu_rtos_wait(npu_inference_req_t *req, uint32_t timeout_ms)
{
    if (req == NULL || !req->_submitted || req->_done)
        return NPU_ERR_BUSY;

    /* Run all layers (this will yield CPU during each layer wait) */
    npu_status_t result = rtos_run_layers(req->model_bin,
                                          req->io_buf_base,
                                          timeout_ms);

    /* Record results */
    req->result = result;
    req->cycles = npu_get_cycle_count();
    req->_done  = 1;

    /* Release hardware */
    s_caller_task = NULL;
    xSemaphoreGive(s_hw_mutex);

    return result;
}

/* ── Passthrough functions (no mutex needed for read-only queries) ── */

void npu_rtos_get_hw_config(npu_hw_config_t *cfg)
{
    npu_get_hw_config(cfg);
}

npu_version_t npu_rtos_get_version(void)
{
    return npu_get_version();
}
