#!/bin/bash
export TF_LOG_DIR="tf/tblogs/$1"
# run a simple program that generates logs for tensorboard
mkdir -p $TF_LOG_DIR
terminate_cluster
./startservers.sh $1
python -m AlexNet.scripts.train --mode $1 --log_dir $TF_LOG_DIR

# start the tensorboard web server. If you have started the webserver on the VM
# a public ip, then you can view Tensorboard on the browser on your workstation
# (not the CloudLab VMs). Navigate to http://<publicip>:6006 on your browser and
# look under "GRAPHS" tab.

# Under the "GRAPHS" tab, use the options on the left to navigate to the "Run" you are interested in.
tensorboard --logdir $TF_LOG_DIR
