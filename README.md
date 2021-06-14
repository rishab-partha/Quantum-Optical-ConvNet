# Quantum-Optical-ConvNet
A repository for the Research Project Constructing a Quantum Optical Convolutional Neural Network (QOCNN) and providing scripts that evaluate future feasibility.

The codebase here is based on https://github.com/mike-fang/imprecise_optical_neural_network. The CNN code was provided by Rohan Bhomwik. Modifications are contained within the subfolder trained_models, and significant parts of mnist.py, optical_nn.py, and train_mnist.py have been written by me. In addition, the subfolder Data and the files ROC Curves.ipynb, t1-t10.pt, and y1-y10.pt are completely new.

If you wish to train/load/run a sample QOCNN, write 'python train_mnist.py' in the command line and execute.

If you wish to load a sample ONN, write 'python mnist.py' in the command line and execute (WARNING: This may take a while.).

Finally, if you wish to run through the process of generating the ROC curves for a sample QOCNN, run through the file 'ROC Curves.ipynb'.