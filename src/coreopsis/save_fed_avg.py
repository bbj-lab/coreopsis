#!/usr/bin/env python3

"""
a version of FedAvg that saves the model after each training round
"""

import pathlib

import flwr as fl
import transformers

from coreopsis.task import set_weights


class SaveFedAvg(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        net: transformers.PreTrainedModel,
        context: fl.common.Context,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.net = net
        self.context = context

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        set_weights(self.net, fl.common.parameters_to_ndarrays(aggregated_parameters))
        self.net.save_pretrained(
            pathlib.Path(self.context.run_config["model-dir"]).expanduser().resolve()
            / f"fms-flw-round-{server_round}"
        )

        return aggregated_parameters, aggregated_metrics
