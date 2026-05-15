#!/usr/bin/env python3

"""
data loader that supports partitioning
"""

import pathlib

import datasets as ds

from cotorra.loader import Loader as CotorraLoader
from cotorra.util import batched_iter


class Loader(CotorraLoader):
    """Supports partitioning data; removes dependency on the datasets `repeat` method,
    which is not supported in earlier versions of the library"""

    def __init__(
        self,
        cfg,
        processed_data_home: pathlib.Path,
        num_partitions: int = 1,
        partition_id: int = 0,
    ):
        super().__init__(cfg=cfg, processed_data_home=processed_data_home)
        self.partition_id = partition_id
        self.num_partitions = num_partitions

        if self.num_partitions > 1:
            self.dataset = ds.DatasetDict(
                {
                    s: self.dataset[s].shard(
                        num_shards=self.num_partitions, index=self.partition_id
                    )
                    for s in self.splits
                }
            )

    def get_train_data(self):
        return ds.Dataset.from_generator(
            batched_iter,
            gen_kwargs={
                "dset": self.dataset[self.splits[0]].shuffle(generator=self.rng),
                "seq_len": self.cfg.max_seq_len,
            },
        ).with_format("torch")


if __name__ == "__main__":
    from coreopsis.trainer import Trainer

    trainer = Trainer()
    self = Loader(cfg=trainer.cfg, processed_data_home=trainer.processed_data_home)
