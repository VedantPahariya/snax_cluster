#!/bin/bash
cd /ssd_scratch/Vedant/snax_cluster/target/snitch_cluster
make CFG_OVERRIDE=cfg/snax_KUL_cluster.hjson bin/snitch_cluster.vlt -j16
