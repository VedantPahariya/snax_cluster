// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DNN Operations Implementation
// Wrappers around existing conv2d, gemm, and activation functions

#include "dnn_ops.h"
#include "dnn_model.h"
#include "snrt.h"
#include <math.h>
#include <string.h>

// ============================================================================
// Conv2D Operation
// ============================================================================

uint32_t dnn_conv2d(const dnn_conv2d_op_t *op) {
    if (!op || !op->ifmap || !op->ofmap || !op->weights) {
        return 1; // Error
    }

    if (snrt_is_compute_core()) {
        uint32_t core_idx = snrt_cluster_core_idx();
        uint32_t num_cores = snrt_cluster_compute_core_num();
        uint32_t total = op->OH * op->OW * op->CO;
        uint32_t elems_per_core = (total + num_cores - 1) / num_cores;
        uint32_t start = core_idx * elems_per_core;
        uint32_t end = (start + elems_per_core < total) ? start + elems_per_core : total;

        for (uint32_t idx = start; idx < end; idx++) {
            uint32_t co = idx % op->CO;
            uint32_t ow = (idx / op->CO) % op->OW;
            uint32_t oh = idx / (op->OW * op->CO);
            double acc = 0.0;

            for (int32_t fh = 0; fh < op->FH; fh++) {
                int32_t ih = (int32_t)oh * op->stride + fh - op->padding;
                if (ih < 0 || ih >= op->IH) {
                    continue;
                }
                for (int32_t fw = 0; fw < op->FW; fw++) {
                    int32_t iw = (int32_t)ow * op->stride + fw - op->padding;
                    if (iw < 0 || iw >= op->IW) {
                        continue;
                    }
                    for (int32_t ci = 0; ci < op->CI; ci++) {
                        uint32_t ifmap_idx = ((uint32_t)ih * op->IW + (uint32_t)iw) * op->CI + ci;
                        uint32_t weight_idx = (((uint32_t)co * op->FH + (uint32_t)fh) * op->FW + (uint32_t)fw) * op->CI + ci;
                        acc += op->ifmap[ifmap_idx] * op->weights[weight_idx];
                    }
                }
            }

            op->ofmap[idx] = acc;
        }
    }

    snrt_cluster_hw_barrier();
    return 0; // Success
}

// ============================================================================
// Sigmoid Activation
// ============================================================================

uint32_t dnn_sigmoid(const dnn_sigmoid_op_t *op) {
    if (!op || !op->input || !op->output || op->size == 0) {
        return 1; // Error
    }

    // Distribute work across compute cores
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t elements_per_core = (op->size + num_cores - 1) / num_cores;

    uint32_t start_idx = core_idx * elements_per_core;
    uint32_t end_idx = (start_idx + elements_per_core < op->size) ?
                       start_idx + elements_per_core : op->size;

    // Element-wise sigmoid. The SNAX math archive provides expf in this flow.
    for (uint32_t i = start_idx; i < end_idx; i++) {
        float x = (float)op->input[i];
        op->output[i] = (double)(1.0f / (1.0f + expf(-x)));
    }

    snrt_cluster_hw_barrier(); // Wait for all cores
    return 0; // Success
}

// ============================================================================
// ReLU Activation
// ============================================================================

uint32_t dnn_relu(const dnn_relu_op_t *op) {
    if (!op || !op->input || !op->output || op->size == 0) {
        return 1; // Error
    }

    // Distribute work across compute cores
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t elements_per_core = (op->size + num_cores - 1) / num_cores;

    uint32_t start_idx = core_idx * elements_per_core;
    uint32_t end_idx = (start_idx + elements_per_core < op->size) ?
                       start_idx + elements_per_core : op->size;

    // Element-wise ReLU: max(0, x)
    for (uint32_t i = start_idx; i < end_idx; i++) {
        op->output[i] = (op->input[i] > 0.0) ? op->input[i] : 0.0;
    }

    snrt_cluster_hw_barrier(); // Wait for all cores
    return 0; // Success
}

// ============================================================================
// MatMul via GEMMX Accelerator
// ============================================================================

uint32_t dnn_matmul_gemmx(const dnn_matmul_gemmx_op_t *op) {
    if (!op || !op->A || !op->B || !op->C) {
        return 1; // Error
    }

    // GEMMX can only be called from core 0
    // Configuration is complex and device-specific
    // For now, we'll use CPU GEMM as fallback
    // In a real implementation, this would call set_gemmx_streamer_csr(),
    // set_gemmx_csr(), set_gemmx_start(), and wait_gemmx_and_streamer()

    // Fallback to CPU GEMM
    dnn_matmul_cpu_op_t cpu_op = {
        .A = op->A,
        .B = op->B,
        .C = op->C,
        .M = op->M,
        .K = op->K,
        .N = op->N,
        .transpose_a = op->transpose_a,
        .transpose_b = op->transpose_b,
        .alpha = op->alpha,
    };

    return dnn_matmul_cpu(&cpu_op);
}

// ============================================================================
// MatMul via CPU GEMM
// ============================================================================

uint32_t dnn_matmul_cpu(const dnn_matmul_cpu_op_t *op) {
    if (!op || !op->A || !op->B || !op->C) {
        return 1; // Error
    }

    if (snrt_is_compute_core()) {
        uint32_t core_idx = snrt_cluster_core_idx();
        uint32_t num_cores = snrt_cluster_compute_core_num();
        uint32_t total = op->M * op->N;
        uint32_t elems_per_core = (total + num_cores - 1) / num_cores;
        uint32_t start = core_idx * elems_per_core;
        uint32_t end = (start + elems_per_core < total) ? start + elems_per_core : total;

        for (uint32_t idx = start; idx < end; idx++) {
            uint32_t m = idx / op->N;
            uint32_t n = idx % op->N;
            double acc = 0.0;

            for (uint32_t k = 0; k < op->K; k++) {
                double a = op->transpose_a ? op->A[k * op->M + m] : op->A[m * op->K + k];
                double b = op->transpose_b ? op->B[n * op->K + k] : op->B[k * op->N + n];
                acc += a * b;
            }

            op->C[idx] = acc + op->alpha * op->C[idx];
        }
    }

    snrt_cluster_hw_barrier();
    return 0; // Success
}

// ============================================================================
// Element-wise Add (Residual Connections)
// ============================================================================

uint32_t dnn_add(const dnn_add_op_t *op) {
    if (!op || !op->input1 || !op->input2 || !op->output || op->size == 0) {
        return 1; // Error
    }

    // Distribute work across compute cores
    uint32_t core_idx = snrt_cluster_core_idx();
    uint32_t num_cores = snrt_cluster_compute_core_num();
    uint32_t elements_per_core = (op->size + num_cores - 1) / num_cores;

    uint32_t start_idx = core_idx * elements_per_core;
    uint32_t end_idx = (start_idx + elements_per_core < op->size) ?
                       start_idx + elements_per_core : op->size;

    // Element-wise addition with scaling
    double s1 = (op->scale1 == 0.0) ? 1.0 : op->scale1;
    double s2 = (op->scale2 == 0.0) ? 1.0 : op->scale2;

    for (uint32_t i = start_idx; i < end_idx; i++) {
        op->output[i] = s1 * op->input1[i] + s2 * op->input2[i];
    }

    snrt_cluster_hw_barrier();
    return 0; // Success
}

// ============================================================================
// Flatten / Reshape Operation
// ============================================================================

uint32_t dnn_flatten(const dnn_flatten_op_t *op) {
    if (!op || !op->input || !op->output) {
        return 1; // Error
    }

    // Simple memcpy - data is already laid out in memory
    // Just need to copy to output location
    memcpy(op->output, op->input, op->total_size * sizeof(double));
    return 0; // Success
}

// ============================================================================
// Performance Monitoring
// ============================================================================

void dnn_perf_start(perf_counter_t *counter) {
    if (!counter) return;

    if (snrt_global_core_idx() == 0) {
        snrt_reset_perf_counter(SNRT_PERF_CNT0);
        snrt_reset_perf_counter(SNRT_PERF_CNT1);
        snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_CYCLES, 0);
        snrt_start_perf_counter(SNRT_PERF_CNT1, SNRT_PERF_CNT_DMA_BUSY, 0);
        counter->timestamp_start = 0;
    }
    snrt_global_barrier();
}

void dnn_perf_stop(perf_counter_t *counter) {
    if (!counter) return;

    snrt_global_barrier();
    if (snrt_global_core_idx() == 0) {
        snrt_stop_perf_counter(SNRT_PERF_CNT0);
        snrt_stop_perf_counter(SNRT_PERF_CNT1);
        counter->cycles = snrt_get_perf_counter(SNRT_PERF_CNT0);
        counter->dma_busy = snrt_get_perf_counter(SNRT_PERF_CNT1);
        counter->timestamp_end = counter->cycles;
    }
}

void dnn_perf_print(const char *layer_name, const perf_counter_t *counter) {
    if (!counter) return;
    // Note: printf may not be available in embedded context
    // This is a placeholder
}

// ============================================================================
// Utility Functions
// ============================================================================

uint32_t dnn_verify_output(const double *actual, const double *golden,
                           uint32_t size, double tolerance) {
    if (!actual || !golden || size == 0) {
        return size; // All mismatched
    }

    uint32_t errors = 0;
    for (uint32_t i = 0; i < size; i++) {
        double diff = fabs(actual[i] - golden[i]);
        double rel_error = diff / (fabs(golden[i]) + 1e-10);
        if (rel_error > tolerance) {
            errors++;
        }
    }
    return errors;
}

double dnn_checksum(const double *data, uint32_t size) {
    if (!data || size == 0) return 0.0;

    double sum = 0.0;
    for (uint32_t i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

void dnn_print_array(const char *name, const double *data, uint32_t print_size) {
    if (!name || !data) return;
    // Placeholder - printf availability depends on runtime
}
