import argparse
import os
import random
import time
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from libs import models
from libs.checkpoint import resume, save_checkpoint
from libs.class_id_map import get_n_classes
from libs.class_weight import get_class_weight, get_pos_weight
from libs.config import get_config
from libs.dataset import ActionSegmentationDataset, collate_fn
from libs.helper import train, validate
from libs.loss_fn import ActionSegmentationLoss, BoundaryRegressionLoss
from libs.optimizer import get_optimizer
from libs.transformer import TempDownSamp, ToTensor

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="train a network for action recognition"
    )
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="a number used to initialize a pseudorandom number generator.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Add --resume option if you start training from checkpoint.",
    )

    return parser.parse_args()

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))

def change_label_score(best_test, train_loss, epoch, cls_acc, edit_score, f1s):

    best_test['train_loss'] = train_loss
    best_test['epoch'] = epoch
    best_test['cls_acc'] = cls_acc
    best_test['edit'] = edit_score
    best_test['f1s@0.1'] = f1s[0]
    best_test['f1s@0.25'] = f1s[1]
    best_test['f1s@0.5'] = f1s[2]
    best_test['f1s@0.75'] = f1s[3]
    best_test['f1s@0.9'] = f1s[4]

def main() -> None:

    start_start = time.time()

    # argparser
    args = get_arguments()

    # configuration
    config = get_config(args.config)

    result_path =  os.path.join(config.result_path, config.dataset, config.model.split('.')[-2], 'split' + str(config.split))

    print('\n---------------------------result_path---------------------------\n')
    print('result_path:',result_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # cpu or cuda
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        device = config.device

    # Dataloader
    # Temporal downsampling is applied to only videos in LARA
    downsamp_rate = 4 if config.dataset == "LARA" else 1

    train_data = ActionSegmentationDataset(
        config.dataset,
        transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
        mode="trainval" if not config.param_search else "training",
        split=config.split,
        dataset_dir=config.dataset_dir,
        csv_dir=config.csv_dir,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True if config.batch_size > 1 else False,
        collate_fn=collate_fn,
    )

    # if you do validation to determine hyperparams
    if config.param_search:
        val_data = ActionSegmentationDataset(
            config.dataset,
            transform=Compose([ToTensor(), TempDownSamp(downsamp_rate)]),
            mode="validation",
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
        )

        val_loader = DataLoader(
            val_data,
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

    optimizer = get_optimizer(
        config.optimizer,
        model,
        config.learning_rate,
        momentum=config.momentum,
        dampening=config.dampening,
        weight_decay=config.weight_decay,
        nesterov=config.nesterov,
    )

    # resume if you want
    columns = ["epoch", "lr", "train_loss"]

    # if you do validation to determine hyperparams
    if config.param_search:
        columns += ["val_loss", "cls_acc", "edit"]
        columns += [
            "f1s@{}".format(config.iou_thresholds[i])
            for i in range(len(config.iou_thresholds))
        ]
        columns += ["bound_acc", "precision", "recall", "bound_f1s"]

    begin_epoch = 0
    best_loss = float("inf")

    # Define temporary variables for evaluation scores
    best_test_acc =  {'epoch':0,'train_loss':0,'cls_acc':0,'edit':0,'f1s@0.1':0,'f1s@0.25':0,'f1s@0.5':0,'f1s@0.75':0,'f1s@0.9':0}
    best_test_F1_10 =  best_test_acc.copy()
    best_test_F1_50 =  best_test_acc.copy()

    log = pd.DataFrame(columns=columns)

    if args.resume:
        if os.path.exists(os.path.join(result_path, "checkpoint.pth")):
            checkpoint = resume(result_path, model, optimizer)
            begin_epoch, model, optimizer, best_loss = checkpoint
            log = pd.read_csv(os.path.join(result_path, "log.csv"))
            print("training will start from {} epoch".format(begin_epoch))
        else:
            print("there is no checkpoint at the result folder")

    # criterion for loss
    if config.class_weight:
        class_weight = get_class_weight(
            config.dataset,
            split=config.split,
            dataset_dir=config.dataset_dir,
            csv_dir=config.csv_dir,
            mode="training" if config.param_search else "trainval",
        )
        class_weight = class_weight.to(device)
    else:
        class_weight = None

    criterion_cls = ActionSegmentationLoss(
        ce=config.ce,
        focal=config.focal,
        tmse=config.tmse,
        gstmse=config.gstmse,
        weight=class_weight,
        ignore_index=255,
        ce_weight=config.ce_weight,
        focal_weight=config.focal_weight,
        tmse_weight=config.tmse_weight,
        gstmse_weight=config.gstmse,
    )

    pos_weight = get_pos_weight(
        dataset=config.dataset,
        split=config.split,
        csv_dir=config.csv_dir,
        mode="training" if config.param_search else "trainval",
    ).to(device)

    criterion_bound = BoundaryRegressionLoss(pos_weight=pos_weight)

    # train and validate model
    print("---------- Start training ----------")

    for epoch in range(begin_epoch, config.max_epoch):
        # training
        start = time.time()

        train_loss = train(
            train_loader,
            model,
            criterion_cls,
            criterion_bound,
            config.lambda_b,
            optimizer,
            device,
        )
        train_time = (time.time() - start) / 60

        # if you do validation to determine hyperparams
        if config.param_search:
            start = time.time()
            (
                val_loss,
                cls_acc,
                edit_score,
                segment_f1s,
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ) = validate(
                val_loader,
                model,
                criterion_cls,
                criterion_bound,
                config.lambda_b,
                device,
                config.dataset,
                config.dataset_dir,
                config.iou_thresholds,
                config.boundary_th,
                config.tolerance,
                config.refinement_method,
            )

            if (epoch >0):
                # save a model if top1 cls_acc is higher than ever
                if best_loss > val_loss:
                    best_loss = val_loss

                if cls_acc > best_test_acc['cls_acc']:
                    change_label_score(best_test_acc, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_acc_model.prm')
                    )

                if segment_f1s[0] > best_test_F1_10['f1s@0.1']:
                    change_label_score(best_test_F1_10, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_F1_0.1_model.prm')
                    )

                if segment_f1s[2] > best_test_F1_50['f1s@0.5']:
                    change_label_score(best_test_F1_50, train_loss, epoch, cls_acc, edit_score, segment_f1s)
                    torch.save(
                        model.state_dict(),
                        os.path.join(result_path, 'best_test_F1_0.5_model.prm')
                    )
 
        # save checkpoint every epoch
        save_checkpoint(result_path, epoch, model, optimizer, best_loss)

        # write logs to dataframe and csv file
        tmp = [epoch, optimizer.param_groups[0]["lr"], train_loss]

        # if you do validation to determine hyperparams
        if config.param_search:
            tmp += [
                val_loss,
                cls_acc,
                edit_score,
            ]
            tmp += segment_f1s
            tmp += [
                bound_acc,
                precision,
                recall,
                bound_f1s,
            ]

        tmp_df = pd.DataFrame(tmp, index=log.columns).T
        log = pd.concat([log, tmp_df], ignore_index=True)
        log.to_csv(os.path.join(result_path, "log.csv"))

        val_time = (time.time() - start) / 60


        eta_time = (config.max_epoch-epoch)*(train_time+val_time)
        if config.param_search:
            # if you do validation to determine hyperparams
            print(
                'epoch: {}  lr: {:.4f}  train_time: {:.2f}min  train loss: {:.4f}  val_time: {:.2f}min  eta_time: {:.2f}min  val loss: {:.4f}  val_acc: {:.2f}  val_F1@0.1: {:.2f}  val_F1@0.5: {:.2f}  b_F10: {:.2f}  b_F50: {:.2f}'
                .format(epoch, optimizer.param_groups[0]['lr'], train_time, train_loss, val_time, eta_time, val_loss, cls_acc, \
                segment_f1s[0], segment_f1s[2], best_test_F1_10['f1s@0.1'], best_test_F1_50['f1s@0.5'])
            )
        else:
            print(
                "epoch: {}\tlr: {:.4f}\ttrain loss: {:.4f}".format(
                    epoch, optimizer.param_groups[0]["lr"], train_loss
                )
            )

    # delete checkpoint
    os.remove(os.path.join(result_path, "checkpoint.pth"))

    print('\n---------------------------best_test_acc---------------------------\n')
    print('{}'.format(best_test_acc))
    print('\n---------------------------best_test_F1_10---------------------------\n')
    print('{}'.format(best_test_F1_10))
    print('\n---------------------------best_test_F1_50---------------------------\n')
    print('{}'.format(best_test_F1_50))
    print('\n---------------------------all_train_time---------------------------\n')
    print('all_train_time: {:.2f}min'.format((time.time() - start_start) / 60))

    best_test_acc = pd.DataFrame.from_dict(best_test_acc, orient='index').T
    best_test_F1_10 = pd.DataFrame.from_dict(best_test_F1_10, orient='index').T
    best_test_F1_50 = pd.DataFrame.from_dict(best_test_F1_50, orient='index').T
    log = pd.concat([log, best_test_acc], ignore_index=True)
    log = pd.concat([log, best_test_F1_10], ignore_index=True)
    log = pd.concat([log, best_test_F1_50], ignore_index=True)
    log.to_csv(os.path.join(result_path, 'log.csv'), index=False)

    print("Done!")


if __name__ == "__main__":
    main()
