import numpy as np
from holdem.utils import action_table, player_table, community_table
from treys import Deck
from sklearn.preprocessing import OneHotEncoder
from encoders import Encoder
from decoders import Decoder

import time

class TrainingEnv():
    def __init__(self, env, other_players, n_seats, encoder=None, decoder=None, debug=False):
        assert len(other_players) == n_seats - 1
        self.env = env
        self.other_players = other_players
        self.n_seats = n_seats
        self._debug = debug
        self._encoder = encoder if not encoder is None else Encoder(n_seats)
        self._decoder = decoder if not decoder is None else Decoder()
        self._add_players(n_seats)

    @property
    def minimum_raise(self):
        return max(self.env._bigblind, self.env._lastraise + self.env._tocall)

    @property
    def maximum_raise(self):
        return self.env._current_player.stack + self.env._current_player.currentbet

    @property
    def pot_size(self):
        return self.env._totalpot

    @property
    def amount_to_call(self):
        return self.env._tocall - self.env._current_player.currentbet

    @property
    def n_observation_dimensions(self):
        return self._encoder.n_dim

    @property
    def n_actions(self):
        return 3 + 23

    def swap_opponent_model(self, new_model):
        for opponent in self.other_players:
            opponent.swap_model(new_model)
    
    def reset(self):
        (player_states, (community_infos, community_cards)) = self._reset()
        encoded_state = self._encoder.encode(player_states, community_infos, community_cards, 0)
        self.outer_start_time = time.time()
        return encoded_state

    def _reset(self):
        for p in self.env._seats:
            p.reset_stack()
        (player_states, (community_infos, community_cards)) = self.env.reset()
        self.start_stacks = []
        players, _ = zip(*player_states)
        for player in players:
            self.start_stacks.append(player[player_table.STACK])
        self.our_seat = players[0][player_table.SEAT_ID]
        self.first_round = True
        self.initial_player_states = player_states
        self.initial_community_infos = community_infos
        self.initial_community_cards = community_cards
        return (player_states, (community_infos, community_cards))
    
    def step(self, action):
        start = time.time()
        outer_duration = time.time() - self.outer_start_time
        our_id = 0
        done = False
        if self.first_round:
            self.first_round = False
            if self._debug:
                print('Starting new round:', self.env._round)
                print("Letting others play after reset...")
            player_states, community_infos, community_cards, reward, done, info =\
                self._other_players_play(self.initial_player_states, self.initial_community_infos, self.initial_community_cards)

        if done:
            if self._debug:
                print('Opponent folded, starting new round...')
            (player_states, (community_infos, community_cards)) = self._reset()
            #player_id = self.env._current_player.player_id
        else:
            #player_id = self.env._current_player.player_id
            move = self._decoder.decode(action, self.n_seats, self.minimum_raise, self.maximum_raise, 
                                        self.pot_size, self.amount_to_call, debug=self._debug)
            actions = np.zeros((self.n_seats, 2))
            actions[our_id,:] = move
            if self._debug:
                print('Action from Training Agent', actions)
                print('Round:', self.env._round)
            (player_states, (community_infos, community_cards)), reward, done, info = self.env.step(actions)
            if self._debug:
                self.render()
                print("Letting others play after step...")
            if not done:
                player_states, community_infos, community_cards, reward, done, info =\
                                self._other_players_play(player_states, community_infos, community_cards, reward, done, info)
        if self._debug:
            print('bb', self.env._bigblind, 
            'last raise', self.env._lastraise, 
            'tocall', self.env._tocall,
            'current bet', self.env._current_player.currentbet,
            'holdem reward', reward,
            'done', done)
        reward = reward[our_id]
        state = self._encoder.encode(player_states, community_infos, community_cards, 0)
        self.outer_start_time = time.time()
        # info['step_duration'] = time.time() - start
        # info['outer_duration'] = outer_duration
        return (state, reward, done, info)
    
    def render(self, mode='human', close=False):
        return self.env.render(mode, close)

    def _other_players_play(self, player_states, community_infos, community_cards, reward=None, done=False, info=None):
        if self._debug:
            print("... others playing now ...")
        # Other players act before training player with seat 0
        to_act_pos = community_infos[community_table.TO_ACT_POS]
        while to_act_pos != self.our_seat and self.env._round < 5 and not done: 
            encoded_state = self._encoder.encode(player_states, community_infos, community_cards, to_act_pos)
            player = self.other_players[to_act_pos - 1]
            move = player.declare_action(player_states, community_infos, community_cards, encoded_state)
            if player.encoded_output:
                move = self._decoder.decode(move, self.n_seats, self.minimum_raise, 
                                    self.maximum_raise, self.pot_size, self.amount_to_call, debug=self._debug)
            actions = np.zeros((self.n_seats, 2), dtype=int)
            actions[to_act_pos,:] = move
            if self._debug:
                print('Actions from Other:', actions, 'Pos:', to_act_pos)
                print('Round:', self.env._round)
            states, reward, done, info = self.env.step(actions.astype(dtype=int))
            player_states, (community_infos, community_cards) = states

            if self._debug:
                self.render()
            to_act_pos = community_infos[community_table.TO_ACT_POS]

        return player_states, community_infos, community_cards, reward, done, info

    def _add_players(self, n_players):
        for i in range(n_players):
            self.env.add_player(i, stack=2000)

    # def _decode_action_for_player(self, action):
    #     player_id = self.env._current_player.player_id
    #     move = self.action_to_move[action]
    #     if self.amount_to_call == 0:
    #         if move in [action_table.CALL, action_table.FOLD]:
    #             move = action_table.CHECK # if np.random.random() < 0.5 else RAISE # Let betting be decided only with action > FOLD
    #     else:
    #         if move == action_table.CHECK:
    #             move = action_table.FOLD
    #     amount = self.minimum_raise
    #     if action > action_table.FOLD:
    #         amount_index = action - action_table.FOLD
    #         if amount_index == 0:
    #             amount = self.minimum_raise
    #         else:
    #             # Relative raise size
    #             amount = self.betsizes[amount_index - 1] * self.pot_size
    #             # Move size to legal range
    #     if move == action_table.RAISE and self.minimum_raise >= self.maximum_raise:
    #         if self.amount_to_call > 0:
    #             move = action_table.CALL
    #         else:
    #             move = action_table.CHECK
    #     amount = max(min(amount, self.maximum_raise), min(self.minimum_raise, self.maximum_raise))
    #     actions = np.zeros((self.n_seats, 2))
    #     actions[player_id,:] = [move, amount]
    #     if self._debug:
    #         print('\n')
    #         print('maxraise', self.maximum_raise,'minraise', self.minimum_raise)
    #         print(move, amount)
    #     return actions

    