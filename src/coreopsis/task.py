#!/usr/bin/env python3

"""
utilities for the server and clients
"""

import collections
import pathlib

import torch
from flwr.common import Context


def get_weights(net):
    return [val.cpu().to(torch.float32).numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = collections.OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    net.tie_weights()


def unpack_context(context: Context):
    return map(
        lambda p: pathlib.Path(p).expanduser().resolve(),
        [
            context.run_config["training-config"],
            context.run_config["processed-data-dir"],
            context.run_config["output-home"],
        ],
    )
