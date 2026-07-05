#!/usr/bin/env python3
# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# DNN Inference Test Data Generator
# Generates random test data and golden outputs for multi-layer inference models

import numpy as np
import argparse
import pathlib
import hjson


np.random.seed(42)


def array_to_cstr(a, fmt=float):
    """Convert array to C-style initialization string"""
    out = '{'
    if fmt == float:
        for el in np.asarray(a).flatten():
            out += '{}, '.format(float(el))
    out = out[:-2] + '}' if len(out) > 1 else '{}'
    return out


def conv2d_forward(ifmap, weights, padding=0, stride=1):
    """NCHW Conv2d forward pass."""
    n, ci, ih, iw = ifmap.shape
    co, _, fh, fw = weights.shape

    padded = np.pad(
        ifmap,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode='constant'
    )
    oh = (ih + 2 * padding - fh) // stride + 1
    ow = (iw + 2 * padding - fw) // stride + 1
    ofmap = np.zeros((n, co, oh, ow), dtype=ifmap.dtype)

    for b in range(n):
        for oc in range(co):
            for y in range(oh):
                y0 = y * stride
                for x in range(ow):
                    x0 = x * stride
                    window = padded[b, :, y0:y0 + fh, x0:x0 + fw]
                    ofmap[b, oc, y, x] = np.sum(window * weights[oc])

    return ofmap


def sigmoid_forward(input_data):
    """Element-wise sigmoid"""
    return 1.0 / (1.0 + np.exp(-input_data))


def matmul_forward(A, B, transpose_a=False, transpose_b=False, alpha=0.0):
    """Matrix multiplication"""
    if transpose_a:
        A = A.T
    if transpose_b:
        B = B.T
    return np.matmul(A, B)


def random_tensor(shape, dtype, low=-1.0, high=1.0):
    """Generate deterministic random test data."""
    return np.random.uniform(low, high, size=shape).astype(dtype)


def generate_test_data(config):
    """Generate test data and compute golden outputs for all layers"""

    prec = config.get('prec', 64)
    dtype = np.float64 if prec == 64 else np.float32
    seed = config.get('data_gen', {}).get('seed', 42)
    np.random.seed(seed)
    input_range = config.get('data_gen', {}).get('input_range', [-1.0, 1.0])
    weight_range = config.get('data_gen', {}).get('weight_range', [-0.5, 0.5])

    # Input dimensions
    ih, iw, ic = (
        config['input']['height'],
        config['input']['width'],
        config['input']['channels']
    )

    # Create random input in NCHW layout.
    input_data = random_tensor((1, ic, ih, iw), dtype, input_range[0], input_range[1])

    data_dict = {}
    current_output = input_data

    # Process each layer
    for layer in config.get('layers', []):
        layer_type = layer.get('type')
        layer_name = layer.get('name')

        if layer_type == 'Conv2d':
            # Conv2d layer
            ci = layer['input_dim']['channels']
            ih = layer['input_dim']['height']
            iw = layer['input_dim']['width']
            co = layer['output_dim']['channels']
            fh = layer['filter']['height']
            fw = layer['filter']['width']
            stride = layer['filter']['stride']
            padding = layer['filter']['padding']

            if current_output.shape != (1, ci, ih, iw):
                current_output = random_tensor(
                    (1, ci, ih, iw), dtype, input_range[0], input_range[1]
                )

            # Generate random weights
            weights = random_tensor((co, ci, fh, fw), dtype,
                                    weight_range[0], weight_range[1])

            # Forward pass
            output = conv2d_forward(current_output, weights, padding, stride)

            # Convert formats and store
            current_output_hwc = current_output.transpose(0, 2, 3, 1)
            output_hwc = output.transpose(0, 2, 3, 1)
            weights_hwc = weights.transpose(0, 2, 3, 1)

            data_dict[f'{layer_name}_ifmap_dram'] = current_output_hwc
            data_dict[f'{layer_name}_weights_dram'] = weights_hwc
            data_dict[f'{layer_name}_ofmap_dram'] = output_hwc
            data_dict[f'{layer_name}_ofmap_golden'] = output_hwc

            current_output = output

        elif layer_type == 'Sigmoid':
            ci = layer['input_dim']['channels']
            ih = layer['input_dim']['height']
            iw = layer['input_dim']['width']

            if current_output.shape != (1, ci, ih, iw):
                current_output = random_tensor(
                    (1, ci, ih, iw), dtype, input_range[0], input_range[1]
                )

            # Sigmoid activation
            output = sigmoid_forward(current_output)

            # Store as flat array
            output_flat = output.flatten()
            data_dict[f'{layer_name}_ifmap'] = current_output.flatten()
            data_dict[f'{layer_name}_ofmap_dram'] = output_flat
            data_dict[f'{layer_name}_ofmap_golden'] = output_flat

            current_output = output

        elif layer_type == 'MatMul':
            # Matrix multiplication
            M = layer['dimensions']['M']
            K = layer['dimensions']['K']
            N = layer['dimensions']['N']

            flat_input = current_output.flatten()
            if flat_input.size >= M * K:
                A = flat_input[:M * K].reshape(M, K).astype(dtype)
            else:
                A = random_tensor((M, K), dtype, input_range[0], input_range[1])

            B = random_tensor((K, N), dtype, weight_range[0], weight_range[1])

            # Forward pass
            C = matmul_forward(A, B, transpose_a=False, transpose_b=False, alpha=0.0)

            data_dict[f'{layer_name}_ifmap'] = A
            data_dict[f'{layer_name}_weights_dram'] = B
            data_dict[f'{layer_name}_ofmap_dram'] = C
            data_dict[f'{layer_name}_ofmap_golden'] = C

            # For next layer, flatten
            current_output = C.reshape(1, 1, M, N)

    return data_dict, dtype


def emit_header_file(file_path, config, data_dict, dtype):
    """Emit C header file with all test data"""

    c_dtype = 'double' if dtype == np.float64 else 'float'

    emit_str = "// Copyright 2024 ETH Zurich and University of Bologna.\n"
    emit_str += "// Licensed under the Apache License, Version 2.0, see LICENSE for details.\n"
    emit_str += "// SPDX-License-Identifier: Apache-2.0\n\n"
    emit_str += "// AUTO-GENERATED TEST DATA HEADER\n"
    emit_str += "// Model: {}\n\n".format(config.get('model_name', 'unknown'))

    emit_str += "#pragma once\n\n"
    emit_str += f"#include <stdint.h>\n"
    emit_str += f"static const {c_dtype} data_prec = 1.0e-10;\n\n"

    # Emit each data array
    for name, data in data_dict.items():
        if isinstance(data, np.ndarray):
            shape = data.shape
            size = data.size
            flat_data = data.flatten()

            # Shape comment
            emit_str += f"// {name}: shape {shape}\n"

            # Emit array declaration
            if len(shape) == 1:
                emit_str += f"static {c_dtype} {name}[{size}] = "
            elif len(shape) == 2:
                emit_str += f"static {c_dtype} {name}[{shape[0]}][{shape[1]}] = "
            elif len(shape) == 3:
                emit_str += f"static {c_dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}] = "
            elif len(shape) == 4:
                emit_str += f"static {c_dtype} {name}[{shape[0]}][{shape[1]}][{shape[2]}][{shape[3]}] = "
            else:
                emit_str += f"static {c_dtype} {name}[{size}] = "

            emit_str += array_to_cstr(flat_data) + ";\n\n"

    with open(file_path, 'w') as f:
        f.write(emit_str)


def main():
    parser = argparse.ArgumentParser(description='Generate DNN test data')
    parser.add_argument(
        "-c",
        "--cfg",
        type=pathlib.Path,
        required=True,
        help='Select param config file'
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default='data.h',
        help='Output header file'
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Load configuration
    with args.cfg.open() as f:
        config = hjson.loads(f.read())

    if args.verbose:
        print(f"Loading config from: {args.cfg}")
        print(f"Model: {config.get('model_name')}")
        print(f"Precision: FP{config.get('prec')}")

    # Generate test data
    data_dict, dtype = generate_test_data(config)

    if args.verbose:
        print(f"Generated {len(data_dict)} data arrays")
        for name in data_dict.keys():
            print(f"  - {name}")

    # Emit header file
    emit_header_file(args.output, config, data_dict, dtype)

    if args.verbose:
        print(f"Wrote test data to: {args.output}")


if __name__ == '__main__':
    main()
