import sys
sys.path.append('core')

import argparse
import torch
from pathlib import Path
from core.raft_stereo import RAFTStereo
from onnxsim import simplify
import onnx
import torch.onnx as thonnx
import os

DEVICE = 'cuda'

def export_onnx(args):
  model = torch.nn.DataParallel(RAFTStereo(args), device_ids=[0])
  model.load_state_dict(torch.load(args.restore_ckpt))
  model = model.module
  model.to(DEVICE)
  model.eval()

  output_directory = Path(args.output_directory)
  output_directory.mkdir(exist_ok=True)

  model_filename = "raft-stereo-sceneflow-520-616.onnx"

  with torch.no_grad():
    sample_input = (torch.zeros(1, 3, 520, 616).to(DEVICE), torch.zeros(1, 3, 520, 616).to(DEVICE), args.valid_iters, None, True)
    thonnx.export(model,                 # model being run
                  sample_input,              # model input (or a tuple for multiple inputs)
                  os.path.join(args.output_directory, model_filename), # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['left', 'right'],   # the model's input names
                  output_names = ['disparity', 'upscaled_disparity'])
    # optional: save a simplified version of the model
    stereo_onnx_model = onnx.load(os.path.join(args.output_directory,model_filename))  # load onnx model
    stereo_model_simp, stereo_check = simplify(stereo_onnx_model)
    assert stereo_check, "Simplified stereo model could not be validated"
    onnx.save(stereo_model_simp, os.path.join(args.output_directory, "simplified-" + model_filename))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--restore_ckpt', help="restore checkpoint", required=True)
  parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
  parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/Middlebury/MiddEval3/testH//im0.png")
  parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/Middlebury/MiddEval3/testH//im1.png")
  parser.add_argument('--output_directory', help="directory to save output", default="/home/ocina/projects/becca/oc_ti/oc_ti/camera/tests/data/")
  parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

  # Architecture choices
  parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
  parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
  parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
  parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
  parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
  parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
  parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
  parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
  parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")

  args = parser.parse_args()

  export_onnx(args)