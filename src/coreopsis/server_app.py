#!/usr/bin/env python3

"""
centralized things the server does
"""

import logging

from flwr.common import Context, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from coreopsis.save_fed_avg import SaveFedAvg
from coreopsis.task import get_weights
from coreopsis.trainer import Trainer


def server_fn(context: Context):

    log(logging.INFO, f"{context.run_config=}")

    net = Trainer().model_init()
    weights = get_weights(net)
    initial_parameters = ndarrays_to_parameters(weights)

    strategy = SaveFedAvg(
        fraction_fit=1.0,  # Fraction of clients used during training
        fraction_evaluate=1.0,  # Fraction of clients used during validation
        initial_parameters=initial_parameters,
        net=net,
        context=context,
        on_fit_config_fn=lambda server_round: {"server_round": server_round},
    )
    config = ServerConfig(num_rounds=context.run_config["num-server-rounds"])

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
