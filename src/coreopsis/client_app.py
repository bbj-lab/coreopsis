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
from transformers import TrainingArguments

from coreopsis.task import get_weights, set_weights, unpack_context
from cotorra.loader import Loader
from cotorra.trainer import Trainer, TrainerWithCustomLoss


class FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        self.dset = json.loads(self.context.run_config["datasets"])[
            self.context.node_config["partition-id"]
        ]
        training_cfg, processed_data_dir, output_home = unpack_context(context)
        self.ct = Trainer(training_cfg, processed_data_dir / self.dset, output_home)
        self.loader = Loader(training_cfg, processed_data_dir / self.dset)

        self.trainer = TrainerWithCustomLoss(
            model=self.ct.model_init(),
            data_collator=self.ct.collate_fn,
            compute_loss_func=self.ct.loss,
            train_dataset=self.ct.loader.get_train_data(),
            eval_dataset=self.ct.loader.get_tuning_data(),
            args=TrainingArguments(
                output_dir=str(output_home), **self.ct.cfg.training_args
            ),
        )

        self.created = time.time()
        self.id = hashlib.md5(
            (f"{id(self)}-{self.created}").encode("utf-8")
        ).hexdigest()[:7]

        self.pid = self.context.node_config["partition-id"]
        log(logging.INFO, f"Client {self.id} initialized (pid={self.pid})")

    def fit(self, parameters, config):
        set_weights(self.trainer.model, parameters)
        num_rounds = int(self.context.run_config["num-server-rounds"])
        round_num = config.get("server_round", 1)
        self.trainer.train_dataset = shard = self.trainer.train_dataset.shard(
            num_shards=num_rounds, index=round_num - 1
        )
        log(
            logging.INFO,
            f"training {self.id} (pid={self.pid}), round {round_num}/{num_rounds}, "
            f"{len(shard)} examples...",
        )
        self.trainer.train()
        return get_weights(self.trainer.model), len(shard), {}

    def evaluate(self, parameters, config):
        set_weights(self.trainer.model, parameters)
        loss = self.trainer.evaluate()["eval_loss"]
        log(logging.INFO, f"Validation {self.id} (pid={self.pid}): {loss=:.3f}")
        return float(loss), len(self.trainer.eval_dataset), {}


def client_fn(context: Context):
    return FlowerClient(context).to_client()


app = ClientApp(client_fn)
