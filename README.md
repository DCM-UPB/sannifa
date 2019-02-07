# sannifa
Simple Artificial Neural Network Interface for Function Approximation

This repository is meant to provide a simple common interface to different ANN implementations (e.g. PyTorch, DCM-UPB/FeedForwardNeuralNetwork), written in C++ and with intended usage for general function approximation problems in scientific computing.

More specifically we are considering a situation where you seek to approximate an unknown (!) function, by optimizing a parametrized neural network model according to some problem-specific cost function. To compute your cost function and respective gradients, you may not only require gradients with respect to the model parameters, but also first or second order derivatives with respect to the model input. For use cases like this we provide a simple and convenient general interface to use in your optimization, completely independent of which ANN-library is used under the hood.

# Requirements
At the moment the build requires PyTorch (https://pytorch.org/cppdocs/installing.html) and DCM-UPB/FeedForwardNeuralNetwork (https://github.com/DCM-UPB/FeedForwardNeuralNetwork). Also we use CMake for building, so you need to have that as well.

# Build
Starting from the project root directory, first copy `config_template.sh` to a new file `config.sh` and change the contained paths appropriately. Afterwards, simply execute `./build.sh`. For details on what is happening look into that script.

# Run
Currently you can run small testing programs used for development (code in `test/`), by invoking `./build/test/test_torch` or `./build/test/test_ffnn` . You can prepend these commands with `OMP_NUM_THREADS=1` to suppress inherent OpenMP multi-threading (in the case of PyTorch).
Furthermore, in `benchmark/` there are simple benchmarks of ANN evaluation with both libffnn and libtorch backend.
