#!/usr/bin/env python3

"""
functional utilities used by the server and clients
"""

import collections
import pathlib

import torch
from transformers import TrainingArguments

from coreopsis.loader import Loader
from coreopsis.trainer import Trainer as c_Trainer
from coreopsis.trainer import TrainerWithCustomLoss


def load_data(num_partitions: int, partition_id: int):
    """partition MIMIC into `num_partitions-1` parts of similar size and load UCMC
    as partition number `num-partitions-1`"""
    if partition_id < num_partitions - 1:
        dataset = Loader(
            cfg=c_Trainer().cfg,
            processed_data_home=pathlib.Path("./processed/mimic").resolve(),
            num_partitions=num_partitions - 1,
            partition_id=partition_id,
        )
    else:
        dataset = Loader(
            cfg=c_Trainer().cfg,
            processed_data_home=pathlib.Path("./processed/ucmc").resolve(),
        )
    return (dataset.get_train_data(), dataset.get_tuning_data())


def train(net, trainloader, testloader):
    trainer = TrainerWithCustomLoss(
        model=net,
        data_collator=(_c := c_Trainer()).collate_fn,
        compute_loss_func=_c.loss,
        train_dataset=trainloader,
        eval_dataset=testloader,
        args=TrainingArguments(output_dir=str(_c.output_home), **_c.cfg.training_args),
    )
    trainer.train()


def test(net, testloader):
    trainer = TrainerWithCustomLoss(
        model=net,
        data_collator=(_c := c_Trainer()).collate_fn,
        compute_loss_func=_c.loss,
        eval_dataset=testloader,
        args=TrainingArguments(output_dir=str(_c.output_home), **_c.cfg.training_args),
    )
    return trainer.evaluate()["eval_loss"]


def get_net():
    return c_Trainer().model_init()


def get_weights(net):
    return [val.cpu().to(torch.float32).numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    net.tie_weights()
