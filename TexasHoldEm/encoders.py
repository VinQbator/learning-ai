import numpy as np
from holdem.utils import action_table, player_table, community_table
from treys import Deck, Card, Evaluator
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Encoder():
    pot_normalized_community = [
        community_table.SMALL_BLIND, 
        community_table.POT,
        community_table.LAST_RAISE,
        community_table.MINRAISE,
        community_table.TO_CALL]

    pot_normalized_player = [
        player_table.STACK, 
        player_table.LAST_SIDEPOT]

    def __init__(self, n_seats, ranking_encoding='norm', concat=True, drop_cards=False, split_cards=False):
        self.n_seats = n_seats
        self.ranking_encoding = ranking_encoding
        self._deck = np.array(Deck.GetFullDeck(), dtype=np.int64)
        self._deck_alt = np.concatenate((np.array([-1], dtype=np.int64), self._deck))
        self._evaluator = Evaluator()
        self.concat = concat

    @property
    def n_card_dim(self):
        return (265 + 104) + (7463 if self.ranking_encoding == 'one-hot' else 1 if self.ranking_encoding == 'norm' else 0)

    @property
    def n_other_dim(self):
        return 6 + self.n_seats + 6 * self.n_seats

    @property
    def n_dim(self):
        return self.n_card_dim + self.n_other_dim

    def encode(self, player_states, community_infos, community_cards, our_seat):
        player_infos, player_hands = zip(*player_states)
        player_infos = np.array(player_infos, dtype=np.float32)
        community_infos = np.array(community_infos, dtype=np.float32)

        n_players = player_infos.shape[0]

        full_stack = community_infos[community_table.BIG_BLIND] * 100
        
        community_infos[Encoder.pot_normalized_community] = community_infos[Encoder.pot_normalized_community] / full_stack
        
        community_infos_t = np.zeros(6 + n_players)
        community_infos_t[:6] = community_infos[community_table.SMALL_BLIND:community_table.TO_ACT_POS]
        community_infos_t[int(6+community_infos[community_table.BUTTON_POS])] = 1

        cards = player_hands[0]
        community_cards_t = np.zeros(5*53)
        player_cards_t = np.zeros(52*2)
        
        community_cards_t[[int(i * 53 + np.where(self._deck_alt == community_cards[i])[0]) for i in range(5)]] = 1
        player_cards_t[[int(i * 52 + np.where(self._deck == int(cards[i]))[0]) for i in range(2)]] = 1

        player_infos[:,Encoder.pot_normalized_player] /= full_stack
        players_info_t = np.zeros((n_players, 6))
        players_info_t[:] = player_infos[:,:player_table.ID]

        if self.ranking_encoding is None:
            hand = []
        else:
            community_cards = [card for card in community_cards if card > 0]
            if len(community_cards) > 0:
                hand_rank = self._evaluator.evaluate(cards, community_cards)
            else:
                hand_rank = -1

            if self.ranking_encoding == 'norm':
                if hand_rank > 0:
                    hand = [1 / hand_rank]
                else:
                    hand = [-1]
            elif self.ranking_encoding == 'one-hot':
                hand = [0] * 7643
                hand[hand_rank] = 1
            else:
                raise Exception('Unknown ranking encoding!')

        hand_t = np.array(hand)

        if self.concat:
            return np.concatenate(
                (community_infos_t.flatten(), players_info_t.flatten(), community_cards_t.flatten(), player_cards_t.flatten(), hand_t.flatten()), 
                axis=0)

        return community_infos_t, players_info_t, community_cards_t, player_cards_t, hand_t