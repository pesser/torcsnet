# TorcsNet

Vision-based autonomous driving in [TORCS](http://torcs.sourceforge.net/).
The controller is implemented as a Convolutional Neural Network with
[Caffe](http://caffe.berkeleyvision.org/). To train the network, driving
data of an expert (either human or bot) is collected into a
[LevelDB](http://leveldb.org/) storage. At run time, TORCS and the network
communicate over shared memory to achieve real-time control of the vehicle
using nothing but visual input. See also the
[presentation](assets/torcsnet_presentation.pdf).

## Demo

![demo](assets/demo.gif)

[HQ](https://gfycat.com/ResponsibleHeavenlyAiredaleterrier)


## Usage

In the build directory, add the following symbolic links:

- `torcs_train_input`: leveldb database containing the (shuffled) input
  frames to train on.
- `torcs_train_target`: leveldb database containing the regression targets
  for `torcs_train_input`.

- `torcs_test_input`: leveldb database containing the frames to test on.
- `torcs_test_target`: leveldb database containing ground-truth of test set.

- `torcs_train_mean.binaryproto`: binaryproto containing the mean of all
  frames (training and test) as produced by caffe's `compute_image_mean`
  tool.
- `torcs_train_normalization.binaryproto`: binaryproto containing parameters
  for a linear transformation that normalizes the regression targets
  (over training and test set) as produced by `normalize.cpp`.

Notice that we calculated the mean and normalization parameter over training
and test set but if you retrain the network from scratch it will also be
fine to use only the training set. Also the parameters in
`network_solver.prototxt` are choosen such that for our training and test
set, we train for a total of 50 epochs and snapshot and test (the whole test
set) after every epoch. You might want to adjust these values if your
datasets have different sizes.

Example usage:

    cd ${BUILD_DIR}
    cmake ${SRC_DIR}
    ln -s ${DATA_DIR}/350000_Training_input_shuffled_train torcs_train_input
    ln -s ${DATA_DIR}/350000_Training_input_shuffled_test torcs_test_input
    ln -s ${DATA_DIR}/350000_Training_target_normalized_shuffled_train torcs_train_target
    ln -s ${DATA_DIR}/350000_Training_target_normalized_shuffled_test torcs_test_target
    ln -s ${DATA_DIR}/350000_Training_input_mean.binaryproto torcs_train_mean.binaryproto
    ln -s ${DATA_DIR}/350000_Training_target_normalization_parameters.binaryproto torcs_train_normalization.binaryproto

    caffe train --solver=network_solver.prototxt
    # or to continue from snapshot use
    # caffe train --solver=network_solver.prototxt --snapshot=network_snapshot_iter_XXX.solverstate

To visualize the performance of a snapshot use

    ./visualize_prediction ${DATA_DIR}/350000_Training_input network_deploy.prototxt network_snapshot_iter_XXX.caffemodel torcs_train_normalization.binaryproto


