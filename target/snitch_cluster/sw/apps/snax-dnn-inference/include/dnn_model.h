// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DNN Model Architecture Definitions
// Defines the structure for multi-layer neural network models

#pragma once

#include <stdint.h>
#include "dnn_ops.h"

// ============================================================================
// Layer Definition
// ============================================================================

#define MAX_LAYERS 10
#define MAX_NAME_LEN 32

typedef struct {
    op_type_t type;
    char name[MAX_NAME_LEN];

    // Dimensions (context-dependent)
    int32_t M, N, K;          // For MatMul: M x K * K x N
    int32_t H, W, C;          // For Conv: height, width, channels
    int32_t OH, OW, OC;       // Output dimensions
    int32_t FH, FW, FC;       // Filter dimensions
    int32_t padding, stride;  // Conv parameters

    // Data pointers
    void *input;
    void *output;
    void *weights;
    void *bias;

    // Flags
    uint8_t use_gemmx;        // For MatMul: use GEMMX (1) or CPU GEMM (0)
    uint8_t transpose_a;      // MatMul: transpose A
    uint8_t transpose_b;      // MatMul: transpose B

    // Precision
    data_precision_t precision;

    // Performance tracking
    perf_counter_t perf;
} dnn_layer_t;

// ============================================================================
// Model Definition
// ============================================================================

typedef struct {
    char name[MAX_NAME_LEN];
    int num_layers;
    dnn_layer_t layers[MAX_LAYERS];

    // Metadata
    int32_t input_height, input_width, input_channels;
    int32_t output_height, output_width, output_channels;
    data_precision_t precision;

    // Global performance metrics
    uint32_t total_cycles;
    uint32_t total_dma_busy;
} dnn_model_t;

// ============================================================================
// Model Construction and Execution
// ============================================================================

/**
 * Initialize model structure
 */
void dnn_model_init(dnn_model_t *model, const char *name);

/**
 * Add layer to model
 * Returns layer index for configuration, or -1 on error
 */
int32_t dnn_model_add_layer(dnn_model_t *model, const dnn_layer_t *layer);

/**
 * Execute all layers in sequence
 * Handles cluster synchronization and performance monitoring
 */
uint32_t dnn_model_execute(dnn_model_t *model);

/**
 * Verify model outputs against golden reference
 * Checks all layers sequentially
 */
uint32_t dnn_model_verify(dnn_model_t *model, const double *golden_outputs,
                          uint32_t output_size);

/**
 * Print model summary and statistics
 */
void dnn_model_print_summary(const dnn_model_t *model);

/**
 * Allocate L1 memory for intermediate layers (distributed across cores)
 */
double *dnn_alloc_l1_buffer(uint32_t size);

/**
 * Free L1 memory
 */
void dnn_free_l1_buffer(double *ptr);

// ============================================================================
// Predefined Model Configurations
// ============================================================================

/**
 * Simple 2-layer model for testing:
 * Conv -> Sigmoid -> MatMul (via GEMMX) -> Conv -> Sigmoid -> MatMul + Add
 *
 * Input: 1x1x128x128
 * Output: 1x16x64x64
 */
void dnn_model_create_2layer_gemmx(dnn_model_t *model);

/**
 * CPU-based version (fallback for non-GEMMX systems)
 */
void dnn_model_create_2layer_cpu(dnn_model_t *model);
