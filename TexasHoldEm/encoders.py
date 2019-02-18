import numpy as np
from holdem.utils import action_table, player_table, community_table
from treys import Deck
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class Encoder():
    pot_normalized_community = [
        community_table.SMALL_BLIND, 
        community_table.POT,
        community_table.LAST_RAISE,
        community_table.MINRAISE,
        community_table.TO_CALL]

    pot_normalized_player = [player_table.STACK, player_table.LAST_SIDEPOT]

    def __init__(self, n_seats, ranking_encoding='one-hot'):
        self.n_seats = n_seats
        self.ranking_encoding = ranking_encoding
        #self._init_transformers()
        self.n_dim = 265 + 104 + 6 + n_seats + 6 * n_seats #+ (7463 if ranking_encoding == 'one-hot' else 1)
        self._deck = np.array(Deck.GetFullDeck(), dtype=np.int64)
        self._deck_alt = np.concatenate((np.array([-1], dtype=np.int64), self._deck))

    def encode(self, player_states, community_infos, community_cards, our_seat, concat=True):
        player_infos, player_hands = zip(*player_states)
        player_infos = np.array(player_infos, dtype=np.float32)
        player_hands = np.array(player_hands, dtype=np.int64)
        community_infos = np.array(community_infos, dtype=np.float32)
        community_cards = np.array(community_cards, dtype=np.int64)

        n_players = player_infos.shape[0]

        full_stack = community_infos[community_table.BIG_BLIND] * 100
        
        community_infos[Encoder.pot_normalized_community] = community_infos[Encoder.pot_normalized_community] / full_stack
        
        community_infos_t = np.zeros(6 + n_players)
        community_infos_t[:6] = community_infos[community_table.SMALL_BLIND:community_table.TO_ACT_POS]
        community_infos_t[int(6+community_infos[community_table.BUTTON_POS])] = 1

        community_cards_t = np.zeros(5*53)
        community_cards_t[[int(i * 53 + np.where(self._deck_alt == community_cards[i])[0]) for i in range(5)]] = 1

        cards = player_hands[0]
        player_cards_t = np.zeros(52*2)
        player_cards_t[[int(i * 52 + np.where(self._deck == int(cards[i]))[0]) for i in range(2)]] = 1

        player_infos[:,Encoder.pot_normalized_player] /= full_stack
        players_info_t = np.zeros((n_players, 6))
        players_info_t[:] = player_infos[:,:]

        if concat:
            return np.concatenate(
                (community_infos_t.flatten(), players_info_t.flatten(), community_cards_t.flatten(), player_cards_t.flatten()), 
                axis=0)

        return community_infos_t, players_info_t, community_cards_t, player_cards_t#, hand_t

        

    def encode_slow(self, player_states, community_infos, community_cards, our_seat):
        player_infos, player_hands = zip(*player_states)
        player_infos = np.array(player_infos, dtype=np.float32)
        player_hands = np.array(player_hands, dtype=np.float32)
        community_infos = np.array(community_infos, dtype=np.float32)
        community_cards = np.array(community_cards, dtype=np.float32)

        player_index = int(community_infos[community_table.TO_ACT_POS])
        n_players = len(player_infos)

        hand = [player_infos[player_index][player_table.HAND_RANK]]
        if self.ranking_encoding == 'one-hot':
            hand = self._handrank_transformer.fit_transform([hand])
        elif self.ranking_encoding == 'norm':
            hand = [[hand[0] / 7642.0]]
        else:
            raise Exception('Unknown ranking encoding!')

        hand_t = np.array(hand)

        full_stack = community_infos[community_table.BIG_BLIND] * 100
        
        community_infos[Encoder.pot_normalized_community] = community_infos[Encoder.pot_normalized_community] / full_stack
        community_infos[community_table.BUTTON_POS] = (community_infos[community_table.BUTTON_POS] + our_seat) % n_players # Shift seats

        community_infos_t = self._community_info_transformer.fit_transform(community_infos.reshape(1, -1))
        community_cards_t = self._community_cards_transformer.fit_transform(community_cards.reshape(1, -1))

        cards = player_hands[player_index]
        player_cards_t = self._player_cards_transformer.fit_transform(cards.reshape(1, -1))

        start = player_infos[:-our_seat] # Shift seats
        end = player_infos[-our_seat:]
        player_infos = np.concatenate((end, start), axis=0)

        players_info_t = np.array([])
        for info in player_infos:
            info[player_table.SEAT_ID] = (info[player_table.SEAT_ID] + our_seat) % n_players # Shift seats
            info[Encoder.pot_normalized_player] = info[Encoder.pot_normalized_player] / full_stack
            players_info_t = np.append(players_info_t, self._player_info_transformer.fit_transform([info]))

        return np.concatenate((community_infos_t.flatten(),
            community_cards_t.flatten(),
            player_cards_t.flatten(),
            players_info_t.flatten(),
            hand_t.flatten()),
            axis=0)

    def _init_transformers(self):
        # WAY slower than doing manually with numpy

        card_columns = [0, 1]
        community_cards_columns = [0, 1, 2, 3, 4]
        position_columns = [community_table.BUTTON_POS]
        drop_positions = [community_table.TO_ACT_POS]
        seat_columns = [player_table.SEAT_ID]
        hand_columns = [player_table.HAND_RANK]
        
        deck = Deck.GetFullDeck()
        extended_deck = [-1, *deck]

        self._player_info_transformer = ColumnTransformer(
            transformers=[(
                'seat',
                OneHotEncoder(categories=np.repeat(np.array([range(self.n_seats)]), len(seat_columns), axis=0), handle_unknown='ignore'), 
                seat_columns),
                ('drop_hand', 'drop', hand_columns)],
            remainder='passthrough')

        self._player_cards_transformer = ColumnTransformer(
            transformers=[(
                'cards', 
                OneHotEncoder(categories=np.repeat(np.array([deck]), 2, axis=0), handle_unknown='ignore', sparse = False), 
                card_columns)],
            remainder='passthrough')

        self._community_info_transformer = ColumnTransformer(
            transformers=[(
                'positions', 
                OneHotEncoder(categories=np.repeat(np.array([range(self.n_seats)]), len(position_columns), axis=0), handle_unknown='ignore'), 
                position_columns),
                ('drop_to_act', 'drop', drop_positions)],
            remainder='passthrough')

        self._community_cards_transformer = ColumnTransformer(
            transformers=[(
                'cards', 
                OneHotEncoder(categories=np.repeat(np.array([extended_deck]), 5, axis=0), handle_unknown='ignore', sparse = False), 
                community_cards_columns)],
            remainder='passthrough')
    
        self._handrank_transformer = OneHotEncoder(
            categories=np.repeat(np.array([[-1, *range(1, 7463)]]), 1, axis=0), 
            handle_unknown='ignore', 
            sparse = False)