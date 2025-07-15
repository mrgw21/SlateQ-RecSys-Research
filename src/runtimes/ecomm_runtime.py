from recsim_ng.lib.tensorflow import runtime

class ECommRuntime(runtime.TFRuntime):
    def __init__(self, story, num_steps):
        super().__init__(network=story, num_steps=num_steps)