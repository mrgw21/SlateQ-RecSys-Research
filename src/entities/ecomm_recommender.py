from recsim_ng.entities.state_models import static

class ECommRecommender(static.StaticStateModel):
    def specs(self):
        return {}

    def initial_state(self):
        return {}