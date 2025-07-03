import argparse
import torch
import torch.nn as nn
from core.raft_stereo import RAFTStereo
import os
import tempfile
from triton_flavor import log_model
from mlflow_utils import get_or_create_experiment
import mlflow
from .export_to_onnx import MODEL_CONFIG

experiment_id = get_or_create_experiment("raft_stereo")
mlflow.set_experiment(experiment_id=experiment_id)


class RaftStereoTraced(nn.Module):
    """
    A clean wrapper for tracing the RAFTStereo model for deployment.
    Its purpose is to call the underlying model in test_mode and extract only the
    final, upscaled disparity map, creating a simple, single-tensor output.
    """
    def __init__(self, underlying_model, valid_iters):
        super().__init__()
        self.model = underlying_model
        self.valid_iters = valid_iters
        self.model.eval()

    def forward(self, left_image, right_image):
        """
        This forward pass expects raw image tensors with values in the [0, 255] range.
        It passes them directly to the underlying model, which handles normalization.
        """
        _, upscaled_disparity = self.model(
            left_image,
            right_image,
            iters=self.valid_iters,
            flow_init=None,
            test_mode=True,
        )
        return upscaled_disparity


def export_final_torchscript(args):
    """
    Loads a RAFTStereo model, wraps it for a clean interface, converts it to
    TorchScript, and logs it to MLflow.
    """
    model_name = args.restore_ckpt.split("/")[-1].split(".")[0].split("-")[-1]

    if model_name in MODEL_CONFIG:
        print(f"Found configuration for '{model_name}'. Applying settings...")
        for key, value in MODEL_CONFIG[model_name].items():
            setattr(args, key, value)
    else:
        raise ValueError(f"Config for '{model_name}' not found.")
    
    pytorch_model = RAFTStereo(args)
    model_dp = torch.nn.DataParallel(pytorch_model, device_ids=[0])
    model_dp.load_state_dict(torch.load(args.restore_ckpt))
    pytorch_model = model_dp.module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pytorch_model.to(device)
    pytorch_model.eval()
    wrapper = RaftStereoTraced(pytorch_model, args.valid_iters)
    wrapper.eval()

    with tempfile.TemporaryDirectory() as output_dir:
        model_save_path = os.path.join(output_dir, "models", "model", "1")
        config_save_path = os.path.join(output_dir, "models", "model")
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(config_save_path, exist_ok=True)

        with torch.no_grad():
            sample_input_left = torch.randint(0, 256, (1, 3, *args.input_size), dtype=torch.float32, device=device)
            sample_input_right = torch.randint(0, 256, (1, 3, *args.input_size), dtype=torch.float32, device=device)
            
            traced_script_module = torch.jit.trace(wrapper, (sample_input_left, sample_input_right))
            traced_script_module.save(os.path.join(model_save_path, "model.pt"))
            print("Successfully exported final TorchScript model.")

        try:
            with open("raft_torchscript.pbtxt", "r") as f:
                config_pbtxt = f.read()
        except FileNotFoundError:
            raise FileNotFoundError("A 'template.pbtxt' file is required in the same directory.")
        
        # Ensure platform is set for TorchScript
        h, w = args.input_size
        config_pbtxt = config_pbtxt.replace("<VAL1>", str(h))
        config_pbtxt = config_pbtxt.replace("<VAL2>", str(w))

        with open(os.path.join(config_save_path, "config.pbtxt"), "w") as f:
            f.write(config_pbtxt)
            
        log_model(
            config_save_path,
            artifact_path="models",
            await_registration_for=10,
        )
        print("Successfully logged final model to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final export pipeline for RAFT-Stereo to TorchScript.")
    parser.add_argument("--restore_ckpt", help="Path to model checkpoint", required=True)
    parser.add_argument("--input_size", type=int, nargs=2, default=(960, 1024), help="Input image size H W")
    parser.add_argument("--valid_iters", type=int, default=16, help="Number of GRU iterations")
    
    # The script will automatically populate other necessary model args from MODEL_CONFIG
    args, _ = parser.parse_known_args()
    args.input_size = tuple(args.input_size)
    
    export_final_torchscript(args)
