#!/bin/bash

source cluster_utils.sh
start_cluster startserver.py cluster
python -m AlexNet.scripts.train --mode cluster
