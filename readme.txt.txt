Group 16: Nick Daly, Maulik Shah, Fayi Zhang, Jinzi Zhang


Run the following command :
$bash run.sh <deploy-mode>

Where <deploy-mode> is either one of the three options : [single, cluster, cluster2].

The run.sh script does the following things :
1. Terminates any existing servers.
2. Calls ./startservers.sh in the required mode.
3. Runs the Alexnet training code in the required mode.
4. Calls tensorboard to create the tf graph and show other resource statistics.


The logs/ directory consists of the runtimes of training Alexnet in both cluster and cluster2 modes. 

The tf/tblogs directory consists of the logging output of tensorflow which goes in as input to tensorboard.
