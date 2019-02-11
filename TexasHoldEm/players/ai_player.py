from players.player_base import PlayerBase
from keras.models import Model
from rl.core import Agent
from rl.agents import DQNAgent
import numpy as np

class AIPlayer(PlayerBase):
    def __init__(self, model, window_length):
        self.swap_model(model)
        self.window_length = window_length

    @property
    def encoded_output(self):
        return True

    def declare_action(self, player_states, community_infos, community_cards, encoded_state):
        # encoded_state = encoded_state.reshape(1, -1)
        # if len(encoded_state) < self.window_length: ### TODO: Will need actual history. Store here maybe....
        #     print(encoded_state)
        #     while len(encoded_state) < self.window_length:
        #         encoded_state = np.concatenate((encoded_state, encoded_state[-1].reshape(1, -1)), axis=0)
        #     print(encoded_state.shape)
        # elif len(encoded_state) > self.window_length:
        #     print(encoded_state)
        #     encoded_state = encoded_state[:self.window_length]
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