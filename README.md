# cfdonnx

A Python module for exporting pre-trained CFD models to ONNX, making them interoperable with other ML frameworks and compatible with browsers.

It currently supports U-Net architecture and PyTorch models, but it will be soon extended to other frameworks and architectures.

Reproducible examples can be found at [openfoam-cfd-rom](https://github.com/simzero/openfoam-ml-rom) usign [DeepCFD](https://github.com/mdribeiro/DeepCFD).

## Installation

The module can be installed with:

```
pip3 install cfdonnx
```

## Usage

```
Usage:  python3 -m cfdonnx [OPTIONS]

Options:
    -n, --net           TEXT  network architecture: UNetEx or AutoEncoder (default: UNetEx)
    -i, --input         PATH  checkpoint (default: checkpoint.pt)
    -o, --output        PATH  ONNX output file (default: checkpoint.onnx)
    -k, --kernel-size   INT   kernel size (optional, read from state_dict['kernel_size] by default )
    -f, --filters       TEXT  filter size, e.g. 8,16,32,32 (optional, read from state_dict['filters'] by default)
    -c  --channels      INT   number of channels (optional, read from state_dict['input_shape'] by default)
    -x  --nx            INT   X dimension (optional, read from state_dict['input_shape'] by default)
    -y  --ny            INT   Y dimension (optional, read from state_dict['input_shape'] by default )
    -o, --output        PATH  Save model path (default: mymodel.pt)

Example:

python3 -m cfdonnx \
        --net UNetEx \
        --input flowAroundObstacles.pt \
        --output flowAroundObstacles.onnx
```

You can use your CFD ONNX models on runtime in Babylon.js as showcased at https://play.simzero.com/#D3SFTH#6 for the [flowAroundObstacles](https://github.com/simzero/openfoam-ml-rom/tree/main/OpenFOAM/incompressible/simpleFoam/flowAroundObstacles) example.

A generic template for using ONNX is also available at https://play.simzero.com/#WIB297#1.
