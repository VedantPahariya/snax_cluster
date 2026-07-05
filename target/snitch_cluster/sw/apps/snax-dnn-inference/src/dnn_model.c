// Copyright 2024 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// DNN Model Implementation
// Model architecture setup and execution control

#include "dnn_model.h"
#include "snrt.h"
#include <string.h>

// ============================================================================
// Model Construction
// ============================================================================

void dnn_model_init(dnn_model_t *model, const char *name) {
    if (!model) return;

    strncpy(model->name, name, MAX_NAME_LEN - 1);
    model->name[MAX_NAME_LEN - 1] = '\0';
    model->num_layers = 0;
    model->total_cycles = 0;
    model->total_dma_busy = 0;
    model->precision = DATA_FP64;

    memset(model->layers, 0, sizeof(model->layers));
}

int32_t dnn_model_add_layer(dnn_model_t *model, const dnn_layer_t *layer) {
    if (!model || !layer || model->num_layers >= MAX_LAYERS) {
        return -1; // Error
    }

    int32_t idx = model->num_layers;
    model->layers[idx] = *layer;
    model->num_layers++;
    return idx;
}

uint32_t dnn_model_execute(dnn_model_t *model) {
    if (!model || model->num_layers == 0) {
        return 1; // Error
    }

    uint32_t total_errors = 0;

    for (int32_t layer_idx = 0; layer_idx < model->num_layers; layer_idx++) {
        dnn_layer_t *layer = &model->layers[layer_idx];

        dnn_perf_start(&layer->perf);

        // Execute operation based on type
        uint32_t errors = 0;
        switch (layer->type) {
        case OP_CONV2D: {
            dnn_conv2d_op_t op = {
                .ifmap = (double *)layer->input,
                .ofmap = (double *)layer->output,
                .weights = (double *)layer->weights,
                .CO = layer->OC,
                .CI = layer->C,
                .IH = layer->H,
                .IW = layer->W,
                .OH = layer->OH,
                .OW = layer->OW,
                .FH = layer->FH,
                .FW = layer->FW,
                .padding = layer->padding,
                .stride = layer->stride,
            };
            errors = dnn_conv2d(&op);
            break;
        }
        case OP_SIGMOID: {
            uint32_t size = layer->H * layer->W * layer->C;
            dnn_sigmoid_op_t op = {
                .input = (double *)layer->input,
                .output = (double *)layer->output,
                .size = size,
            };
            errors = dnn_sigmoid(&op);
            break;
        }
        case OP_RELU: {
            uint32_t size = layer->H * layer->W * layer->C;
            dnn_relu_op_t op = {
                .input = (double *)layer->input,
                .output = (double *)layer->output,
                .size = size,
            };
            errors = dnn_relu(&op);
            break;
        }
        case OP_MATMUL_GEMMX: {
            dnn_matmul_gemmx_op_t op = {
                .A = (double *)layer->input,
                .B = (double *)layer->weights,
                .C = (double *)layer->output,
                .M = layer->M,
                .K = layer->K,
                .N = layer->N,
                .transpose_a = layer->transpose_a,
                .transpose_b = layer->transpose_b,
                .alpha = 0.0,
            };
            errors = dnn_matmul_gemmx(&op);
            break;
        }
        case OP_MATMUL_CPU: {
            dnn_matmul_cpu_op_t op = {
                .A = (double *)layer->input,
                .B = (double *)layer->weights,
                .C = (double *)layer->output,
                .M = layer->M,
                .K = layer->K,
                .N = layer->N,
                .transpose_a = layer->transpose_a,
                .transpose_b = layer->transpose_b,
                .alpha = 0.0,
            };
            errors = dnn_matmul_cpu(&op);
            break;
        }
        case OP_ADD: {
            uint32_t size = layer->H * layer->W * layer->C;
            dnn_add_op_t op = {
                .input1 = (double *)layer->input,
                .input2 = (double *)layer->weights,
                .output = (double *)layer->output,
                .size = size,
                .scale1 = 1.0,
                .scale2 = 1.0,
            };
            errors = dnn_add(&op);
            break;
        }
        case OP_FLATTEN: {
            dnn_flatten_op_t op = {
                .input = (double *)layer->input,
                .output = (double *)layer->output,
                .total_size = layer->H * layer->W * layer->C,
            };
            errors = dnn_flatten(&op);
            break;
        }
        default:
            errors = 1;
            break;
        }

        dnn_perf_stop(&layer->perf);
        total_errors += errors;
    }

    return total_errors;
}

uint32_t dnn_model_verify(dnn_model_t *model, const double *golden_outputs,
                          uint32_t output_size) {
    if (!model || !golden_outputs) {
        return 1;
    }

    // Get last layer's output
    dnn_layer_t *last_layer = &model->layers[model->num_layers - 1];
    double *actual = (double *)last_layer->output;

    // Verify with 1% tolerance
    return dnn_verify_output(actual, golden_outputs, output_size, 0.01);
}

void dnn_model_print_summary(const dnn_model_t *model) {
    if (!model) return;
    // Placeholder for summary printing
}

double *dnn_alloc_l1_buffer(uint32_t size) {
    // Allocate from L1 TCDM using snrt_l1_next()
    double *ptr = (double *)snrt_l1_next();
    if (!ptr) return NULL;

    // Advance pointer for next allocation
    snrt_l1_next();
    return ptr;
}

void dnn_free_l1_buffer(double *ptr) {
    // L1 memory doesn't need explicit freeing in this model
    // It's automatically managed per cluster execution
}

// ============================================================================
// Predefined Model: 2-Layer GEMMX
// ============================================================================

void dnn_model_create_2layer_gemmx(dnn_model_t *model) {
    dnn_model_init(model, "2layer_gemmx");

    // Model architecture:
    // Conv1 (7x7) -> Sigmoid -> MatMul (GEMMX)
    // -> Conv2 (3x3) -> Sigmoid -> MatMul (GEMMX) + Add

    model->input_height = 128;
    model->input_width = 128;
    model->input_channels = 1;
    model->output_height = 64;
    model->output_width = 64;
    model->output_channels = 16;

    // Layer 0: Conv2d (7x7 kernel, stride=2)
    // Input: 1x1x128x128 -> Output: 1x16x64x64
    {
        dnn_layer_t layer = {
            .type = OP_CONV2D,
            .name = "conv1",
            .H = 128,
            .W = 128,
            .C = 1,
            .OH = 64,
            .OW = 64,
            .OC = 16,
            .FH = 7,
            .FW = 7,
            .padding = 3,
            .stride = 2,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 1: Sigmoid activation
    {
        dnn_layer_t layer = {
            .type = OP_SIGMOID,
            .name = "sigmoid1",
            .H = 64,
            .W = 64,
            .C = 16,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 2: MatMul (GEMMX) - 64x16 * 16x64 = 64x64
    {
        dnn_layer_t layer = {
            .type = OP_MATMUL_GEMMX,
            .name = "matmul1_gemmx",
            .M = 64,
            .K = 16,
            .N = 64,
            .use_gemmx = 1,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 3: Conv2d (3x3 kernel, stride=1)
    // Input: 1x16x64x64 -> Output: 1x16x64x64
    {
        dnn_layer_t layer = {
            .type = OP_CONV2D,
            .name = "conv2",
            .H = 64,
            .W = 64,
            .C = 16,
            .OH = 64,
            .OW = 64,
            .OC = 16,
            .FH = 3,
            .FW = 3,
            .padding = 1,
            .stride = 1,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 4: Sigmoid activation
    {
        dnn_layer_t layer = {
            .type = OP_SIGMOID,
            .name = "sigmoid2",
            .H = 64,
            .W = 64,
            .C = 16,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 5: MatMul (GEMMX) - 64x16 * 16x64 = 64x64
    {
        dnn_layer_t layer = {
            .type = OP_MATMUL_GEMMX,
            .name = "matmul2_gemmx",
            .M = 64,
            .K = 16,
            .N = 64,
            .use_gemmx = 1,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }
}

// ============================================================================
// Predefined Model: 2-Layer CPU (Fallback)
// ============================================================================

void dnn_model_create_2layer_cpu(dnn_model_t *model) {
    dnn_model_init(model, "2layer_cpu");

    // Same architecture but using CPU GEMM instead of GEMMX
    // (Implementation is identical except for OP_MATMUL_CPU vs OP_MATMUL_GEMMX)

    model->input_height = 128;
    model->input_width = 128;
    model->input_channels = 1;
    model->output_height = 64;
    model->output_width = 64;
    model->output_channels = 16;

    // Layer 0: Conv2d
    {
        dnn_layer_t layer = {
            .type = OP_CONV2D,
            .name = "conv1",
            .H = 128,
            .W = 128,
            .C = 1,
            .OH = 64,
            .OW = 64,
            .OC = 16,
            .FH = 7,
            .FW = 7,
            .padding = 3,
            .stride = 2,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 1: Sigmoid
    {
        dnn_layer_t layer = {
            .type = OP_SIGMOID,
            .name = "sigmoid1",
            .H = 64,
            .W = 64,
            .C = 16,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 2: MatMul (CPU)
    {
        dnn_layer_t layer = {
            .type = OP_MATMUL_CPU,
            .name = "matmul1_cpu",
            .M = 64,
            .K = 16,
            .N = 64,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 3: Conv2d
    {
        dnn_layer_t layer = {
            .type = OP_CONV2D,
            .name = "conv2",
            .H = 64,
            .W = 64,
            .C = 16,
            .OH = 64,
            .OW = 64,
            .OC = 16,
            .FH = 3,
            .FW = 3,
            .padding = 1,
            .stride = 1,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 4: Sigmoid
    {
        dnn_layer_t layer = {
            .type = OP_SIGMOID,
            .name = "sigmoid2",
            .H = 64,
            .W = 64,
            .C = 16,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }

    // Layer 5: MatMul (CPU)
    {
        dnn_layer_t layer = {
            .type = OP_MATMUL_CPU,
            .name = "matmul2_cpu",
            .M = 64,
            .K = 16,
            .N = 64,
            .precision = DATA_FP64,
        };
        dnn_model_add_layer(model, &layer);
    }
}
