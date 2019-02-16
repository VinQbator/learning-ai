import numpy as np

from players.player_base import PlayerBase
from holdem.utils import action_table, player_table, community_table

class RandomPlayer(PlayerBase):
    @property
    def encoded_output(self):
        return True

    @property
    def encoded_input(self):
        return False

    def declare_action(self, player_states, community_infos, community_cards, encoded_state):
        return np.random.random_integers(0, 25)