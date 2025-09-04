import copy
import logging
import os
import os.path
import pickle
import sys
import time

import numpy as np
import torch

from metrics.results import ContinualLearningEvaluator
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    device = device.split(",")

    print("Seeds:", seed_list)
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def hhmmss(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _train(args):
    torch.set_float32_matmul_precision("medium")

    if args["model_name"] in [
        "InfLoRA",
        "InfLoRA_domain",
        "InfLoRAb5_domain",
        "InfLoRAb5",
        "InfLoRA_CA",
        "InfLoRA_CA1",
    ]:
        logdir = "logs/{}/{}_{}_{}/{}/{}/{}/{}_{}-{}".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["net_type"],
            args["model_name"],
            args["optim"],
            args["rank"],
            args["lamb"],
            args["lame"],
            args["lrate"],
        )
    else:
        logdir = "logs/{}/{}_{}_{}/{}/{}".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["net_type"],
            args["model_name"],
            args["optim"],
        )
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, "{}".format(args["seed"]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    if not os.path.exists(logfilename):
        os.makedirs(logfilename)
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)

    num_classes = len(data_manager._class_order)
    num_tasks = data_manager.nb_tasks
    model._max_classes = num_classes
    cl_evaluator = ContinualLearningEvaluator(
        num_tasks=num_tasks, num_classes=num_classes
    )

    class_schedule = np.split(np.arange(0, num_classes), num_tasks)
    task_durations: list[float] = []

    for train_task_idx in range(data_manager.nb_tasks):
        time_start = time.time()
        logging.info("-" * 80)
        logging.info(
            "Training on task {}/{}".format(train_task_idx + 1, data_manager.nb_tasks)
        )
        logging.info("Classes: {}".format(class_schedule[train_task_idx]))
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)

        logging.info("Evaluating...")
        for test_task_idx in range(data_manager.nb_tasks):
            test_classes = class_schedule[test_task_idx]
            test_task_dataset = data_manager.get_dataset(
                test_classes, source="test", mode="test"
            )
            test_task_loader = torch.utils.data.DataLoader(
                test_task_dataset,
                batch_size=args["batch_size"],
                shuffle=False,
                num_workers=args["num_workers"],
                pin_memory=True,
            )
            model.eval_task(
                train_task_idx, test_task_idx, test_task_loader, cl_evaluator
            )

        model.after_task()

        torch.save(
            model._network.state_dict(),
            os.path.join(logfilename, "task_{}.pth".format(int(train_task_idx))),
        )

        # Track task duration
        task_duration = time.time() - time_start
        task_durations.append(task_duration)

        # Estimate remaining time
        avg_duration = float(np.mean(task_durations))
        time_spent = float(np.sum(task_durations))
        remaining_time = avg_duration * (data_manager.nb_tasks - train_task_idx - 1)
        expected_duration = time_spent + remaining_time
        logging.info("Durations (hh:mm:ss):")
        logging.info(f"  Task:       {hhmmss(task_duration)}")
        logging.info(f"  Total:      {hhmmss(time_spent)}")
        logging.info(f"  Remaining:  {hhmmss(remaining_time)}")
        logging.info(f"  Expected:   {hhmmss(expected_duration)}")

    record = cl_evaluator.result()
    with open(os.path.join(logfilename, "record.pkl"), "wb") as f:
        pickle.dump(record, f)

    for key, value in record.items():
        if isinstance(value, float):
            logging.info("{}: {:.4f}".format(key, value))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(args):
    torch.manual_seed(args["seed"])
    torch.cuda.manual_seed(args["seed"])
    torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
