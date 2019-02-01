# sannifa
Simple Artificial Neural Network Interface for Function Approximation

This repository is meant to provide a simple common interface to different ANN implementations (e.g. tensorflow, DCM-UPB/FeedForwardNeuralNetwork), written in C++ and with intended usage for general function approximation problems in scientific computing.

# Build
Starting from the project root directory, create a build directory (e.g. "build"), enter the directory and invoke "cmake .." . Afterwards, you can compile the library and associated programs via "make".

# Current plan
1) interface base class -> done
2) DCM-UPB/FFNN interface
3) tensorflow interface
4) tests and benchmarks
