class PlayerBase:
    @property
    def encoded_output(self):
        raise NotImplementedError()

    def declare_action(self, player_states, community_infos, community_cards, encoded_state):
        raise NotImplementedError()

    def swap_model(self, model):
        raise NotImplementedError()