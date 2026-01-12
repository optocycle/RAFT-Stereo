import argparse
import torch
from core.raft_stereo import RAFTStereo
from onnxsim import simplify
import onnx
import os
import tempfile
from triton_flavor import log_model
from mlflow_utils import get_or_create_experiment
import mlflow

experiment_id = get_or_create_experiment("raft_stereo")
mlflow.set_experiment(experiment_id=experiment_id)

MODEL_CONFIG = {
    "middlebury": {
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


def export_onnx(args):

    model_name = (
        args.restore_ckpt.split("/")[-1].split(".")[0].split("-")[-1]
    )  # ckpt is named raft-<dataset_name>.pth
    if model_name in MODEL_CONFIG:
        for key, value in MODEL_CONFIG[model_name].items():
            setattr(args, key, value)
    else:
        raise ValueError("Invalid model name. Choose from: middlebury, realtime, eth3d")

    model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.restore_ckpt, map_location=device))
    model = model.module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with tempfile.TemporaryDirectory() as output_dir:
        model_save_path = os.path.join(
            output_dir,
            "models",
            "model",
            "1",
        )
        config_save_path = os.path.join(output_dir, "models", "model")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(config_save_path, exist_ok=True)

        with torch.no_grad():
            sample_input = (
                torch.zeros(1, 3, *args.input_size).to(device),
                torch.zeros(1, 3, *args.input_size).to(device),
                args.valid_iters,
                None,
                True,
            )
            torch.onnx.export(
                model,  # model being run
                sample_input,  # model input (or a tuple for multiple inputs)
                os.path.join(
                    model_save_path, "model.onnx"
                ),  # where to save the model (can be a file or file-like object)
                export_params=True,  # store the trained parameter weights inside the model file
                opset_version=16,  # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                input_names=["left", "right"],  # the model's input names
                output_names=["disparity", "upscaled_disparity"],
                dynamic_axes={"left": {0: "batch_size"}, "right": {0: "batch_size"}},
            )
        # optional: save a simplified version of the model
        if args.simplify:
            stereo_onnx_model = onnx.load(
                os.path.join(model_save_path, "model.onnx")
            )  # load onnx model

            stereo_model_simp, stereo_check = simplify(stereo_onnx_model)
            assert stereo_check, "Simplified stereo model could not be validated"
            onnx.save(stereo_model_simp, os.path.join(model_save_path, "model.onnx"))
        with open("raft.pbtxt", "r") as f:
            config_pbtxt = f.read()
        config_pbtxt = config_pbtxt.replace("<VAL1>", str(args.input_size[0]))
        config_pbtxt = config_pbtxt.replace("<VAL2>", str(args.input_size[1]))
        with open(os.path.join(config_save_path, "config.pbtxt"), "w") as f:
            f.write(config_pbtxt)

        log_model(
            config_save_path,
            artifact_path="models",
            await_registration_for=10,
        )
        mlflow.log_params(args.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", help="restore checkpoint", required=True)
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
        "--simplify", type=bool, help="simplify the onnx model", default=True
    )
    parser.add_argument(
        "--input_size",
        type=tuple,
        default=(960, 1024),
    )
    args = parser.parse_args()

    export_onnx(args)
