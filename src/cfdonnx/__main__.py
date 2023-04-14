import os
import json
import torch
import pickle
import getopt
import sys
import onnx
import onnxruntime as ort
from onnx import ModelProto, StringStringEntryProto
from onnx import shape_inference
from onnx import checker
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable


def parseOpts(argv):
    channels = ""
    nx = ""
    ny = ""
    kernel_size = ""
    filters = ""
    net = ""
    checkpoint = "checkpoint.pt"
    output = "checkpoint.onnx"
    metadata = {}

    try:
        opts, args = getopt.getopt(
                argv,"hc:,x:,y:k:f:n:i:o:m:",
            [
                "channels=",
                "nx=",
                "ny=",
                "kernel-size=",
                "filters=",
                "net=",
                "input=",
                "output=",
                "metadata="
            ]
        )
    except getopt.GetoptError as e:
        print("Invalid option. See python -m cfdonnx --help")
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h' or opt == '--help' or opt == '-help':
            print("cfdonnx "
                "\n    -c  --channels  number of channels (optional, read from state_dict['input_shape'] by default)"
                "\n    -x  --nx  X dimension (optional, read from state_dict['input_shape'] by default)"
                "\n    -y  --ny  Y dimension (optional, read from state_dict['input_shape'] by default )"
                "\n    -k  --kernel-size  kernel size (optional, read from state_dict['kernel_size] by default )"
                "\n    -f  --filters  filter size, e.g. 8,16,32,32 (optional, read from state_dict['filters'] by default )"
                "\n    -n  --net  Network architecture UNetEx or "
                    "AutoEncoder (default: UNetEx)"
                "\n    -i  --input Checkpoint (default: checkpoint.pt)"
                "\n    -o  --output  ONNX output file (default: checkpoint.onnx)"
                "\n    -m  --metadata  Metadata (default: none)"
                "\n    -h  --help  Display help"
                
            )
            sys.exit()
        elif opt in ("-c", "--channels"):
            channels = int(arg)
        elif opt in ("-x", "--nx"):
            nx = int(arg)
        elif opt in ("-y", "--ny"):
            ny = int(arg)
        elif opt in ("-k", "--kernel-size"):
            kernel_size = int(arg)
        elif opt in ("-f", "--filters"):
            filters = [int(x) for x in arg.split(',')]
        elif opt in ("-n", "--net"):
            print("net: ", arg)
            net = arg
        elif opt in ("-i", "--input"):
            checkpoint = arg
        elif opt in ("-o", "--output"):
            output = arg
        elif opt in ("-m", "--metadata"):
            metadata = arg

    options = {
        'channels': channels,
        'nx': nx,
        'ny': ny,
        'kernel_size': kernel_size,
        'filters': filters,
        'net': net,
        'checkpoint': checkpoint,
        'output': output,
        'metadata': metadata,
    }

    return options


def main():
    options = parseOpts(sys.argv[1:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_extension = os.path.splitext(options["checkpoint"])[1]

    if file_extension == '.pt':
        state_dict = torch.load(options["checkpoint"], map_location=device)

        net = None
        net_name = None
        kernel_size = None
        filters = None
        dummy_input = None

        if not options["net"]:
            net_name = state_dict["architecture"]
        else:
            net_name = options["net"]

        if (net_name == "UNetEx"):
            from .models.torch.UNetEx import UNetEx
            net = UNetEx
        else:
            print(net_name + " architecture not supported")
            
        if not options["kernel_size"]:
            kernel_size = state_dict["kernel_size"]
        else:
            kernel_size = options["kernel_size"]

        if not options["filters"]:
            filters = state_dict["filters"]
        else:
            filters = options["filters"]

        if not options["channels"] and not options["nx"] and not options["ny"]:
            input_shape = state_dict["input_shape"]
            dummy_input = torch.randn(input_shape).to(device)
        else:
            dummy_input = torch.randn(
                1,
                options["channels"],
                options["nx"],
                options["ny"]).to(device)

        model = net(
            3,
            3,
            filters=filters,
            kernel_size=kernel_size,
            batch_norm=False,
            weight_norm=False
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(device)

        torch.onnx.export(
           model, 
           dummy_input,
           options["output"],
           input_names = ["inputModel"],
           export_params=True,
           output_names = ["outputModel"],
        )

    else:
       print(fileExtension + " files are not supported yet")

    ort_session = ort.InferenceSession(options["output"])

    model = onnx.load(options["output"])
    checker.check_model(model)

    metadata_dict = json.loads(options["metadata"])

    for key, value in metadata_dict.items():
        meta = StringStringEntryProto()
        meta.key = key
        meta.value = value
        meta = model.metadata_props.add().CopyFrom(meta)

    onnx.save(model, options["output"])


if __name__ == "__main__":
    main()
