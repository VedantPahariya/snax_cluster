// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DNN Operations Library - Centralized wrapper for Conv2d, Sigmoid, MatMul (GEMMX), Add
// Provides unified interface for multi-layer inference

#pragma once

#include <stdint.h>
#include "snrt.h"

// ============================================================================
// Data Types and Enums
// ============================================================================

typedef enum {
    OP_CONV2D,
    OP_SIGMOID,
    OP_RELU,
    OP_MATMUL_GEMMX,
    OP_MATMUL_CPU,
    OP_ADD,
    OP_FLATTEN,
    OP_NONE
} op_type_t;

typedef enum {
    DATA_FP64,
    DATA_FP32,
    DATA_FP16,
    DATA_INT8
} data_precision_t;

// ============================================================================
// Conv2D Operation
// ============================================================================

typedef struct {
    double *ifmap;
    double *ofmap;
    double *weights;
    int32_t CO, CI, IH, IW, OH, OW, FH, FW;
    int32_t padding;
    int32_t stride;
} dnn_conv2d_op_t;

/**
 * Execute Conv2D layer
 * Input: ifmap [IH][IW][CI]
 * Weights: [CO][CI][FH][FW]
 * Output: ofmap [OH][OW][CO]
 */
uint32_t dnn_conv2d(const dnn_conv2d_op_t *op);

// ============================================================================
// Sigmoid Activation
// ============================================================================

typedef struct {
    double *input;
    double *output;
    uint32_t size;  // Total elements
} dnn_sigmoid_op_t;

/**
 * Element-wise Sigmoid: output = 1.0 / (1.0 + exp(-input))
 */
uint32_t dnn_sigmoid(const dnn_sigmoid_op_t *op);

// ============================================================================
// ReLU Activation
// ============================================================================

typedef struct {
    double *input;
    double *output;
    uint32_t size;  // Total elements
} dnn_relu_op_t;

/**
 * Element-wise ReLU: output = max(0, input)
 */
uint32_t dnn_relu(const dnn_relu_op_t *op);

// ============================================================================
// MatMul via GEMMX Accelerator
// ============================================================================

typedef struct {
    double *A;      // M x K matrix
    double *B;      // K x N matrix
    double *C;      // M x N matrix (output)
    int32_t M, K, N;
    uint8_t transpose_a;
    uint8_t transpose_b;
    double alpha;   // Scaling factor
} dnn_matmul_gemmx_op_t;

/**
 * Matrix multiplication using GEMMX hardware accelerator
 * C = A * B + alpha * C (if alpha != 0)
 * Supports transposed inputs
 */
uint32_t dnn_matmul_gemmx(const dnn_matmul_gemmx_op_t *op);

// ============================================================================
// MatMul via CPU GEMM (Fallback)
// ============================================================================

typedef struct {
    double *A;      // M x K matrix
    double *B;      // K x N matrix
    double *C;      // M x N matrix (output)
    int32_t M, K, N;
    uint8_t transpose_a;
    uint8_t transpose_b;
    double alpha;
} dnn_matmul_cpu_op_t;

/**
 * Matrix multiplication using optimized CPU GEMM
 * Fallback when GEMMX not available
 */
uint32_t dnn_matmul_cpu(const dnn_matmul_cpu_op_t *op);

// ============================================================================
// Element-wise Add (Residual Connections)
// ============================================================================

typedef struct {
    double *input1;
    double *input2;
    double *output;
    uint32_t size;  // Total elements
    double scale1;  // Scaling for input1 (default 1.0)
    double scale2;  // Scaling for input2 (default 1.0)
} dnn_add_op_t;

/**
 * Element-wise addition: output = scale1*input1 + scale2*input2
 * Useful for residual connections and feature fusion
 */
uint32_t dnn_add(const dnn_add_op_t *op);

// ============================================================================
// Flatten / Reshape Operation
// ============================================================================

typedef struct {
    double *input;
    double *output;
    uint32_t total_size;  // Must be same for input and output
} dnn_flatten_op_t;

/**
 * In-place reshape from (H,W,C) to 1D array
 * Mostly useful for transitioning from spatial to linear layers
 */
uint32_t dnn_flatten(const dnn_flatten_op_t *op);

// ============================================================================
// Performance Monitoring
// ============================================================================

typedef struct {
    uint32_t cycles;
    uint32_t dma_busy;
    uint64_t timestamp_start;
    uint64_t timestamp_end;
} perf_counter_t;

/**
 * Start performance monitoring
 */
void dnn_perf_start(perf_counter_t *counter);

/**
 * Stop performance monitoring and record metrics
 */
void dnn_perf_stop(perf_counter_t *counter);

/**
 * Print performance metrics
 */
void dnn_perf_print(const char *layer_name, const perf_counter_t *counter);

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Verify output against golden reference
 * Returns number of mismatches
 */
uint32_t dnn_verify_output(const double *actual, const double *golden,
                          uint32_t size, double tolerance);

/**
 * Compute checksum for quick verification
 */
double dnn_checksum(const double *data, uint32_t size);

/**
 * Print first N elements of array (for debugging)
 */
void dnn_print_array(const char *name, const double *data, uint32_t print_size);
