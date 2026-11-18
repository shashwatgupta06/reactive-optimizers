"""Blows the weights up: the loss turns non-finite and the run must recover."""
from example_helper import Base


class Updater(Base):
    def step(self, grads, update, ctx=None):
        for name, p in self.params.items():
            p.data.mul_(float("inf") if update % 7 == 0 else 1.0)


def build(named_params, previous=None, spec=None):
    return Updater(named_params)
