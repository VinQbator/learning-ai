from players.player_base import PlayerBase
from keras.models import Model
from rl.core import Agent
from rl.agents import DQNAgent
import numpy as np

class AIPlayer(PlayerBase):
    def __init__(self, model):
        self.swap_model(model)

    @property
    def encoded_output(self):
        return True

    def declare_action(self, player_states, community_infos, community_cards, encoded_state):
        action = self.predict(encoded_state)
        return action

    def swap_model(self, model):
        self.model = model
        if isinstance(model, Model):
            self.predict = model.predict
        elif isinstance(model, Agent):
            self.predict = model.forward
        else:
            raise Exception("Not sure what method to call to predict")