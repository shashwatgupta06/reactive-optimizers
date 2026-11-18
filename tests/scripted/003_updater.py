"""Raises on its 5th update: must be rolled back inside the trial window."""
import example_helper  # a helper module next to updater.py


class Updater(example_helper.Base):
    def step(self, grads, update, ctx=None):
        self.n += 1
        if self.n == 5:
            raise RuntimeError("deliberate failure")
        for name, p in self.params.items():
            p.data.add_(grads[name], alpha=-0.01)


def build(named_params, previous=None, spec=None):
    return Updater(named_params)
