#!/bin/bash

source cluster_utils.sh
# start_cluster startserver.py single
# start_cluster startserver.py cluster
start_cluster startserver.py $1
