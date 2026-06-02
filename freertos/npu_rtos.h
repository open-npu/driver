/*
 * Open-NPU FreeRTOS Driver — Public API
 * npu_rtos.h — FreeRTOS wrapper around baremetal driver
 *
 * Provides:
 *   - Mutex-protected hardware access (multi-task safe)
 *   - IRQ-driven completion via xTaskNotify (no busy-wait)
 *   - Synchronous blocking and async submit/wait APIs
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NPU_RTOS_H
#define NPU_RTOS_H

#include <stdint.h>
#include <stddef.h>
#include "../baremetal/npu_driver.h"

/* ═══════════════════════════════════════════════════════════════════
 *  FreeRTOS header inclusion
 *
 *  When building without a real FreeRTOS SDK (compile check / unit test),
 *  define NPU_RTOS_STUB_FREERTOS to use minimal type stubs.
 * ═══════════════════════════════════════════════════════════════════ */
#ifdef NPU_RTOS_STUB_FREERTOS

/* Minimal FreeRTOS type stubs for compile checking */
typedef int32_t  BaseType_t;
typedef uint32_t TickType_t;
typedef void    *SemaphoreHandle_t;
typedef void    *TaskHandle_t;

#define pdTRUE           1
#define pdFALSE          0
#define pdPASS           1
#define pdFAIL           0
#define pdMS_TO_TICKS(x) ((TickType_t)(x))
#define portMAX_DELAY    0xFFFFFFFFU
#define eSetValueWithOverwrite 0

static inline SemaphoreHandle_t xSemaphoreCreateMutex(void) { return (SemaphoreHandle_t)1; }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t s, TickType_t t) { (void)s; (void)t; return pdPASS; }
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t s) { (void)s; return pdPASS; }
static inline void vSemaphoreDelete(SemaphoreHandle_t s) { (void)s; }
static inline TaskHandle_t xTaskGetCurrentTaskHandle(void) { return (TaskHandle_t)1; }
static inline BaseType_t xTaskNotifyFromISR(TaskHandle_t t, uint32_t v, int a, BaseType_t *w)
    { (void)t; (void)v; (void)a; *w = pdFALSE; return pdPASS; }
static inline BaseType_t xTaskNotifyWait(uint32_t b, uint32_t c, uint32_t *v, TickType_t t)
    { (void)b; (void)c; (void)v; (void)t; return pdPASS; }
static inline void portYIELD_FROM_ISR(BaseType_t w) { (void)w; }

#else

#include "FreeRTOS.h"
#include "semphr.h"
#include "task.h"

#endif /* NPU_RTOS_STUB_FREERTOS */

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════
 *  Configuration
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    uint32_t default_timeout_ms;   /* default per-layer timeout (0 = infinite) */
} npu_rtos_config_t;

/* ═══════════════════════════════════════════════════════════════════
 *  Async inference request
 * ═══════════════════════════════════════════════════════════════════ */
typedef struct {
    /* Input (caller fills) */
    const void *model_bin;         /* pointer to NPU1 binary */
    uint32_t    ext_mem_base;      /* external memory base address */
    uint32_t    io_buf_base;       /* I/O ping-pong buffer base */

    /* Output (set by driver) */
    npu_status_t result;           /* completion status */
    uint32_t     cycles;           /* total inference cycles (perf counter) */

    /* Internal (do not touch) */
    TaskHandle_t _caller_task;
    volatile uint8_t _submitted;
    volatile uint8_t _done;
} npu_inference_req_t;

/* ═══════════════════════════════════════════════════════════════════
 *  API Functions
 * ═══════════════════════════════════════════════════════════════════ */

/*
 * Initialize FreeRTOS NPU driver.
 * Creates mutex, registers IRQ callback, enables IRQ.
 * Must be called once before any other npu_rtos_* function.
 * Returns NPU_OK on success, NPU_ERR_BUSY if already initialized.
 */
npu_status_t npu_rtos_init(const npu_rtos_config_t *config);

/*
 * Deinitialize FreeRTOS NPU driver.
 * Disables IRQ, deletes mutex, resets state.
 */
void npu_rtos_deinit(void);

/*
 * Run a complete model (synchronous, blocking).
 * Takes mutex, programs each layer, waits for completion via task notification,
 * releases mutex. The calling task sleeps (yields CPU) during inference.
 *
 * timeout_ms: per-layer timeout in milliseconds (0 = use default from config).
 * Returns: NPU_OK, NPU_ERR_TIMEOUT, NPU_ERR_HW_ERROR, NPU_ERR_BAD_MODEL.
 */
npu_status_t npu_rtos_run_model(const void *model_bin,
                                uint32_t ext_mem_base,
                                uint32_t io_buf_base,
                                uint32_t timeout_ms);

/*
 * Submit model for async execution (non-blocking).
 * Takes mutex, starts inference, returns immediately.
 * Caller must call npu_rtos_wait() to wait for completion and release mutex.
 *
 * req: must remain valid until npu_rtos_wait() returns.
 * Returns: NPU_OK on successful submission.
 */
npu_status_t npu_rtos_run_model_async(npu_inference_req_t *req);

/*
 * Wait for async inference to complete.
 * Blocks until all layers finish or timeout.
 * Releases the hardware mutex upon return.
 *
 * timeout_ms: total timeout in milliseconds (0 = infinite).
 * Returns: NPU_OK, NPU_ERR_TIMEOUT, NPU_ERR_HW_ERROR.
 */
npu_status_t npu_rtos_wait(npu_inference_req_t *req, uint32_t timeout_ms);

/*
 * Query hardware configuration (passthrough, no mutex needed).
 */
void npu_rtos_get_hw_config(npu_hw_config_t *cfg);

/*
 * Query hardware version (passthrough, no mutex needed).
 */
npu_version_t npu_rtos_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* NPU_RTOS_H */
