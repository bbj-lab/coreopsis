#!/usr/bin/env python3

"""
distributed things clients do;
NB: new clients are created at the start of every round
"""

import hashlib
import json
import logging
import time

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log

from coreopsis.task import get_weights, set_weights, unpack_context
from cotorra.loader import Loader
from cotorra.trainer import Trainer


class FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        self.dset = json.loads(self.context.run_config["datasets"])[
            self.context.node_config["partition-id"]
        ]
        training_cfg, processed_data_dir, output_home = unpack_context(context)
        self.loader = Loader(training_cfg, processed_data_dir / self.dset)
        self.ct = Trainer(training_cfg, processed_data_dir / self.dset, output_home)
        self.ct.trainer.args.lr_scheduler_type = "constant"
        self.init_lr = self.ct.trainer.args.learning_rate

        self.created = time.time()
        self.id = hashlib.md5(
            (f"{id(self)}-{self.created}").encode("utf-8")
        ).hexdigest()[:7]

        self.pid = self.context.node_config["partition-id"]
        log(logging.INFO, f"Client {self.id} initialized (pid={self.pid})")

    def fit(self, parameters, config):
        set_weights(self.ct.trainer.model, parameters)
        num_rounds = int(self.context.run_config["num-server-rounds"])
        round_num = config.get("server_round", 1)
        progress = (round_num - 1) / max(num_rounds, 1)
        self.ct.trainer.args.learning_rate = (
            self.init_lr * (1 - progress) + 1e-5 * progress
        )
        self.ct.trainer.train_dataset = shard = self.ct.trainer.train_dataset.shard(
            num_shards=num_rounds, index=round_num - 1
        )
        log(
            logging.INFO,
            f"training {self.id} (pid={self.pid}), "
            f"round {round_num}/{num_rounds}, "
            f"lr {self.ct.trainer.args.learning_rate},"
            f"{len(shard)} examples...",
        )
        self.ct.trainer.train()
        return get_weights(self.ct.trainer.model), len(shard), {}

    def evaluate(self, parameters, config):
        set_weights(self.ct.trainer.model, parameters)
        loss = self.ct.trainer.evaluate()["eval_loss"]
        log(logging.INFO, f"Validation {self.id} (pid={self.pid}): {loss=:.3f}")
        return float(loss), len(self.ct.trainer.eval_dataset), {}


def client_fn(context: Context):
    return FlowerClient(context).to_client()


app = ClientApp(client_fn)
