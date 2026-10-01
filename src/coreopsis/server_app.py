#!/usr/bin/env python3

"""
centralized things the server does
"""

import gc
import json
import logging

import torch
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

import coreopsis.save_model_strategy as save_strategy
from coreopsis.task import get_weights, unpack_context
from cotorra.trainer import Trainer


def server_fn(context: Context):

    log(logging.INFO, f"{context.run_config=}")

    dset = json.loads(context.run_config["datasets"]).pop()
    training_cfg, processed_data_dir, output_home = unpack_context(context)
    ct = Trainer(training_cfg, processed_data_dir / dset, output_home)
    net = ct.model_init()
    initial_parameters = ndarrays_to_parameters(get_weights(net))

    # cotorra's Trainer and hf's hold each other; without breaking the cycle the
    # model hf placed on the gpu in its __init__ stays there for the whole run,
    # squatting on memory the clients need when they share the device
    ct.trainer.data_collator = ct.trainer.model_init = None
    ct.trainer.model = ct.trainer.model_wrapped = None
    del ct
    gc.collect()
    torch.cuda.empty_cache()

    fed_strategy = context.run_config.get("fed-strategy", "FedAvg")
    fraction_fit = context.run_config.get("fraction-fit", 1.0)
    strategy = getattr(save_strategy, f"Save{fed_strategy}")(
        fraction_fit=fraction_fit,
        fraction_evaluate=context.run_config.get("fraction-evaluate", 1.0),
        initial_parameters=initial_parameters,
        net=net,
        context=context,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
        **(
            {"server_learning_rate": 1.0, "server_momentum": 0.5}
            if fed_strategy == "FedAvgM"
            else {}
        ),
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
