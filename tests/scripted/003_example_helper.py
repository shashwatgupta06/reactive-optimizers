class Base:
    def __init__(self, named_params):
        self.params, self.n = dict(named_params), 0

    def state_dict(self):
        return {"n": self.n}

    def load_state_dict(self, state):
        self.n = state["n"]
