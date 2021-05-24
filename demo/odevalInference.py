import sys
import os
import time

import argparse

from mega_core.config import cfg
from demo.predictor import VIDDemo

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Visualization")
    parser.add_argument(
        "method",
        choices=["base", "dff", "fgfa", "rdn", "mega"],
        default="base",
        type=str,
        help="which method to use",
    )
    parser.add_argument(
        "config",
        default="configs/vid_R_101_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "checkpoint",
        default="R_101.pth",
        help="The path to the checkpoint for test.",
    )
    parser.add_argument(
        "--config-base",
        default="configs/BASE_RCNN_1gpu.yaml",
        # metavar="FILE",
        help="path to base config file",
    )
    parser.add_argument(
        "--visualize-path",
        default="datasets/ILSVRC2015/Data/VID/val/ILSVRC2015_val_00003001",
        # default="datasets/ILSVRC2015/Data/VID/snippets/val/ILSVRC2015_val_00003001.mp4",
        help="the folder or a video to visualize.",
    )
    parser.add_argument(
        "--suffix",
        default=".jpg",
        help="the suffix of the images in the image folder.",
    )
    parser.add_argument(
        "--output-folder",
        default="demo/visualization/base",
        help="where to store the visulization result.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="if True, input a video for visualization.",
    )
    parser.add_argument(
        "--output-video",
        action="store_true",
        help="if True, output a video.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--confidence-threshold",
        default=0.7,
        type=float,
        help="confidence threshold",
    )
    parser.add_argument(
        "--max-num-frames",
        default=-1,
        type=int,
        help="max number of frames to be processed. -1 means no limits",
    )

    args = parser.parse_args()
    cfg.merge_from_file(args.config_base)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["MODEL.WEIGHT", args.checkpoint])
    cfg.merge_from_list(["MODEL.DEVICE", args.device])

    vid_demo = VIDDemo(
        cfg,
        method=args.method,
        confidence_threshold=args.confidence_threshold,
        output_folder=args.output_folder
    )

    visualization_results = vid_demo.run_on_image_folder_VID(args.visualize_path, suffix=args.suffix, max_num_frames=args.max_num_frames)

    # if not args.video:
    #     visualization_results = vid_demo.run_on_image_folder(args.visualize_path, suffix=args.suffix)
    # else:
    #     visualization_results = vid_demo.run_on_video(args.visualize_path)

    # if not args.output_video:
    #     vid_demo.generate_images(visualization_results)
    # else:
    #     vid_demo.generate_video(visualization_results)

    pass

def run_inference(config_base_file,
                  method,
                  config_file,
                  checkpoint_file,
                  input_images_folder,
                  output_sfx,
                  confidence_threshold,
                  output_folder=None,
                  file_suffix='.JPEG',
                  max_num_frames=-1,
                  ):

    if output_folder is None:
        output_root = '../results/inference/vid_R_50_C4_FGFA_1x'
        time_format = '%Y-%m-%d_%H.%M.%S'
        time_str = '{}'.format(time.strftime(time_format))
        output_folder = os.path.join(output_root, '{}{}'.format(time_str, output_sfx))

    device = 'cuda'
    # device = 'cpu'

    sys.argv = sys.argv[0:1]  # clear previous argv
    sys.argv.append(method)
    sys.argv.append(config_file)
    sys.argv.append(checkpoint_file)
    sys.argv.append('--config-base')
    sys.argv.append(config_base_file)
    sys.argv.append('--visualize-path')
    sys.argv.append(input_images_folder)
    sys.argv.append('--suffix')
    sys.argv.append(file_suffix)
    sys.argv.append('--output-folder')
    sys.argv.append(output_folder)
    sys.argv.append('--output-video')
    sys.argv.append('--device')
    sys.argv.append(device)
    sys.argv.append('--confidence-threshold')
    sys.argv.append(str(confidence_threshold))
    sys.argv.append('--max-num-frames')
    sys.argv.append(str(max_num_frames))

    # run inference
    main()