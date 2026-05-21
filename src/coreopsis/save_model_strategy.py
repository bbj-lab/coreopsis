#!/usr/bin/env python3

"""
a version of FedAvg that saves the model after each training round
"""

import flwr as fl
import transformers

from coreopsis.task import set_weights, unpack_context


class SaveModelMixin(fl.server.strategy.Strategy):
    """
    Mixin to save a copy of the model after each aggregation round;
    can be combined with any fl.server.strategy
    """

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
        *_, output_home = unpack_context(self.context)
        self.net.save_pretrained(output_home / f"coreopsis-round-{server_round}")

        return aggregated_parameters, aggregated_metrics


class SaveFedAvg(SaveModelMixin, fl.server.strategy.FedAvg):
    pass


class SaveFedProx(SaveModelMixin, fl.server.strategy.FedProx):
    pass


class SaveFedAvgM(SaveModelMixin, fl.server.strategy.FedAvgM):
    pass
