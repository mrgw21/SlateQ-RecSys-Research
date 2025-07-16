from recsim_ng.lib.tensorflow import runtime
from tf_agents.drivers import dynamic_step_driver

class ECommRuntime(runtime.TFRuntime):
    def __init__(self, network):
        super().__init__(network=network)

    def execute_with_rl(self, policy, num_steps=1000):
        driver = dynamic_step_driver.DynamicStepDriver(self, policy)
        driver.run(num_steps=num_steps)
        return self.current_value()

    def trajectory(self, num_steps=1000):
        return super().trajectory(length=num_steps)