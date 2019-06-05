from abc import ABCMeta, abstractmethod
from typing import List


class Experiment(ABCMeta):
    @abstractmethod
    def run(
            self,
            lr_list: List[float],
            use_comet: bool,
    ):
        pass
