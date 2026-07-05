# Copyright 2024 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Makefile for SNAX DNN Inference Kernel
# Multi-layer neural network inference application

APP = snax-dnn-inference

# Include directories
INCDIRS = include

# Source files
SRCS = src/main.c src/dnn_model.c include/dnn_layers.c

# Data generation
DATA_CFG = src/params.hjson
DATA_H = src/data.h

# Generate data header from config
$(DATA_H): ./datagen.py $(DATA_CFG)
	./datagen.py -c $(DATA_CFG) -o $@

# Include common build rules
include ../common.mk

# Make data header a dependency
$(DEP): $(DATA_H)

.PHONY: clean-data clean

clean-data:
	rm -f $(DATA_H)

clean: clean-data
