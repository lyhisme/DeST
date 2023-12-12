import argparse
import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.class_id_map import get_n_classes
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import evaluate
from libs.transformer import TempDownSamp, ToTensor
import sys


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="evaluation for action segment refinement network."
    )
    parser.add_argument("config", type=str, help="path to a config file")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="""
            path to the trained model.
            If you do not specify, the trained model,
            'final_model.prm' in result directory will be used.
            """,
    )
    parser.add_argument(
        "--refinement_method",
        type=str,
        default="refinement_with_boundary",
        choices=["refinement_with_boundary", "relabeling", "smoothing"],
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Add --cpu option if you use cpu."
    )

    return parser.parse_args()

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def main():
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path =  os.path.join(config.result_path, config.dataset, config.model.split('.')[-2], 'split' + str(config.split))

    # cpu or gpu
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
            device = config.device

    # Dataloader
    downsamp_rate = 4 if config.dataset == "LARA" else 1

    data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="test",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    loader = DataLoader(
        data,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )

    # load model
    print("---------- Loading Model ----------")

    n_classes = get_n_classes(config.dataset, dataset_dir=config.dataset_dir)
    
    Model = import_class(config.model)
    model = Model(
        in_channel=config.in_channel,
        n_features=config.n_features,
        n_classes=n_classes,
        n_stages=config.n_stages,
        n_layers=config.n_layers,
        n_refine_layers=config.n_refine_layers,
        n_stages_asb=config.n_stages_asb,
        n_stages_brb=config.n_stages_brb,
        SFI_layer=config.SFI_layer,
        dataset=config.dataset,
    )

    # send the model to cuda/cpu
    model.to(device)

    # load the state dict of the model
    if args.model is not None:
        state_dict = torch.load(args.model)
    else:
        # state_dict = torch.load(os.path.join(result_path, "best_test_model.prm")) 
        state_dict = torch.load(os.path.join(result_path, "best_test_model.prm"), map_location={"cuda:0":"cuda:0"})
    model.load_state_dict(state_dict, False)

    # train and validate model
    print("---------- Start testing ----------")

    # evaluation
    evaluate(
        loader,
        model,
        device,
        config.boundary_th,
        config.dataset,
        config.dataset_dir,
        config.iou_thresholds,
        config.tolerance,
        result_path,
        config,  
        args.refinement_method,
    )

    print("Done")

if __name__ == "__main__":
    main()
