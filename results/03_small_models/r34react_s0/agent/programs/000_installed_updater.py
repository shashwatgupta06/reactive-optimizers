"""Starting optimizer program: the training script's own optimizer and schedule, unchanged.

Until this file is edited, the run trains exactly as the baseline does.
"""


class Updater:
    def __init__(self, spec):
        self.opt = spec["user_optimizer"]      # torch.optim.SGD built by the training script
        self.lr_factor = spec["user_lr_factor"]  # update index -> multiplier of each group's base rate
        self.base_lrs = list(spec["base_lrs"])
        self.param_groups = self.opt.param_groups

    def step(self, grads, update, ctx=None):
        f = self.lr_factor(update)
        for group, base in zip(self.opt.param_groups, self.base_lrs):
            group["lr"] = base * f
        self.opt.step()

    def state_dict(self):
        return {"opt": self.opt.state_dict()}

    def load_state_dict(self, state):
        self.opt.load_state_dict(state["opt"])


def build(named_params, previous=None, spec=None):
    return Updater(spec)
