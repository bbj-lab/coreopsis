#!/usr/bin/env python3

"""
distributed things clients do;
NB: new clients are created at the start of every round
"""

import gc
import hashlib
import json
import logging
import math
import time

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log

from coreopsis.task import get_weights, set_weights, unpack_context
from cotorra.trainer import Trainer


class FlowerClient(NumPyClient):
    def __init__(self, context: Context):
        self.context = context
        dsets = json.loads(self.context.run_config["datasets"])
        self.pid = self.context.node_config["partition-id"]
        self.dset = dsets[self.pid]
        training_cfg, processed_data_dir, output_home = unpack_context(context)
        self.ct = Trainer(training_cfg, processed_data_dir / self.dset, output_home)
        self.ct.trainer.model_init = None

        self.created = time.time()
        self.id = hashlib.md5(
            (f"{id(self)}-{self.created}").encode("utf-8")
        ).hexdigest()[:7]

        log(logging.INFO, f"Client {self.id} initialized (pid={self.pid})")

    def _release(self):
        """
        hand this client's model back to the allocator before the next one
        loads; two things outlive the client and have to be cut by hand:
        (1) cotorra's Trainer and hf's hold each other, since hf keeps
            cotorra's bound `collate_fn`/`model_init` as its
            `data_collator`/`model_init`, and
        (2) accelerate's `GradientState` keeps its `_shared_state` on the
            *class*, and `train()` leaves this round's dataloader in its
            `dataloader_references`; that dataloader's collate_fn is the same
            bound method, so a process-global pins the whole graph (weights,
            grads, optimizer state) for the rest of the run
        one `ClientAppActor` serves every client when `num-gpus` is the whole
        device, so skipping either one fills the gpu within a round or two
        """
        trainer = self.ct.trainer
        # guarded: this runs in a `finally`, so a version drift in accelerate
        # should cost us memory, not the round's result
        if (accelerator := getattr(trainer, "accelerator", None)) is not None:
            accelerator.free_memory()
            if (gs := getattr(accelerator, "gradient_state", None)) is not None:
                gs.active_dataloader = None  # writes through to _shared_state
                gs.dataloader_references = [None]  # per GradientState.__init__
        trainer.data_collator = None
        trainer.optimizer = trainer.lr_scheduler = None
        trainer.model = trainer.model_wrapped = None
        trainer.callback_handler = None
        self.ct = trainer = None
        gc.collect()
        torch.cuda.empty_cache()
        log(
            logging.INFO,
            f"released {self.id} (pid={self.pid}), "
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB still allocated",
        )

    def fit(self, parameters, config):
        set_weights(self.ct.trainer.model, parameters)
        num_rounds = int(self.context.run_config["num-server-rounds"])
        round_num = config.get("server_round", 1)
        progress = (round_num - 1) / max(num_rounds, 1)
        self.ct.trainer.args.learning_rate *= 0.5 * (1 + math.cos(math.pi * progress))
        self.ct.trainer.train_dataset = shard = self.ct.trainer.train_dataset.shard(
            num_shards=num_rounds, index=round_num - 1
        )
        log(
            logging.INFO,
            f"training {self.id} (pid={self.pid}), "
            f"round {round_num}/{num_rounds}, "
            f"lr {self.ct.trainer.args.learning_rate}, "
            f"{len(shard)} examples...",
        )
        try:
            self.ct.trainer.train()
            return get_weights(self.ct.trainer.model), len(shard), {}
        finally:
            self._release()

    def evaluate(self, parameters, config):
        set_weights(self.ct.trainer.model, parameters)
        try:
            loss = self.ct.trainer.evaluate()["eval_loss"]
            log(logging.INFO, f"Validation {self.id} (pid={self.pid}): {loss=:.3f}")
            return float(loss), len(self.ct.trainer.eval_dataset), {}
        finally:
            self._release()


def client_fn(context: Context):
    return FlowerClient(context).to_client()


app = ClientApp(client_fn)
