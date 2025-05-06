import argparse
import torch
from pathlib import Path
from core.raft_stereo import RAFTStereo
from onnxsim import simplify
import onnx
import os


def export_onnx(args):
    model_configs = {
        "middlebury": {
            "hidden_dims": [128, 128, 128],
            "context_norm": "batch",
            "n_downsample": 2,
            "corr_levels": 4,
            "corr_radius": 4,
            "n_gru_layers": 3,
            "shared_backbone": True,
            "slow_fast_gru": False,
            "corr_implementation": "reg",
            "mixed_precision": False,
        },
        "eth3d": {
            "hidden_dims": [128, 128, 128],
            "context_norm": "batch",
            "n_downsample": 2,
            "corr_levels": 4,
            "corr_radius": 4,
            "n_gru_layers": 3,
            "shared_backbone": False,
            "slow_fast_gru": False,
            "corr_implementation": "reg",
            "mixed_precision": False,
        },
        "realtime": {
            "hidden_dims": [128, 128, 128],
            "context_norm": "batch",
            "n_downsample": 3,
            "corr_levels": 4,
            "corr_radius": 4,
            "n_gru_layers": 2,
            "shared_backbone": True,
            "slow_fast_gru": True,
            "corr_implementation": "reg",
            "mixed_precision": True,
        },
    }

    if args.model_name in model_configs:
        for key, value in model_configs[args.model_name].items():
            setattr(args, key, value)
    else:
        raise ValueError("Invalid model name. Choose from: middlebury, realtime, eth3d")

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))
    model = model.module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    model_filename = "raft-stereo-sceneflow-520-616.onnx"

    with torch.no_grad():
        sample_input = (
            torch.zeros(1, 3, 520, 616).to(device),
            torch.zeros(1, 3, 520, 616).to(device),
            args.valid_iters,
            None,
            True,
        )
        torch.onnx.export(
            model,  # model being run
            sample_input,  # model input (or a tuple for multiple inputs)
            os.path.join(
                args.output_directory, model_filename
            ),  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=16,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=["left", "right"],  # the model's input names
            output_names=["disparity", "upscaled_disparity"],
        )
        # optional: save a simplified version of the model
        stereo_onnx_model = onnx.load(
            os.path.join(args.output_directory, model_filename)
        )  # load onnx model
        stereo_model_simp, stereo_check = simplify(stereo_onnx_model)
        assert stereo_check, "Simplified stereo model could not be validated"
        onnx.save(
            stereo_model_simp,
            os.path.join(args.output_directory, "simplified-" + model_filename),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", required=True)
    parser.add_argument(
        "--output_directory", help="directory to save output", default="out"
    )
    parser.add_argument(
        "--valid_iters",
        type=int,
        default=64,
        help="number of flow-field updates during forward pass",
    )
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--model_name", type=str, default="middlebury", help="model name"
    )
    # Architecture choices (this is fixed by the model)

    args = parser.parse_args()

    export_onnx(args)
