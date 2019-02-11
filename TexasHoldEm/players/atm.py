from players.player_base import PlayerBase
from holdem.utils import action_table, player_table, community_table

class ATM(PlayerBase):
    @property
    def encoded_output(self):
        return False

    def declare_action(self, player_states, community_infos, community_cards, encoded_state):
        to_call = community_infos[community_table.TO_CALL]
        if to_call > 0:
            return [action_table.CALL, 0]
        return [action_table.CHECK, 0]
