# sannifa
Simple Artificial Neural Network Interface for Function Approximation

This repository is meant to provide a simple common interface to different ANN implementations (e.g. PyTorch, DCM-UPB/FeedForwardNeuralNetwork), written in C++ and with intended usage for general function approximation problems in scientific computing.

# Requirements
At the moment a full build requires PyTorch (https://pytorch.org/cppdocs/installing.html) and DCM-UPB/FeedForwardNeuralNetwork (https://github.com/DCM-UPB/FeedForwardNeuralNetwork). If more interfaces get added it will be made possible to build only for a selection of ANN libraries.

# Build
Starting from the project root directory, first copy "config_template.sh" to a new file "config.sh" and change the contained paths appropriately. Afterwards, simply execute "./build.sh". For details on what is happening look into that script.

# Current plan
1) interface base class -> done
2) DCM-UPB/FFNN interface -> done
3) PyTorch interface
4) tests and benchmarks
