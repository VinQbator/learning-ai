import numpy as np
import gym

from holdem.utils import action_table, player_table, community_table
from treys import Deck
from sklearn.preprocessing import OneHotEncoder
from encoders import Encoder
from decoders import Decoder

import time

# Max betsize in simulation environment (shouldn't really matter with discrete relative to pot sizing)
MAX_BET = 100000

class TrainingEnv():
    def __init__(self, env, other_players, n_seats, encoding='norm', encoder=None, decoder=None, debug=False):
        assert len(other_players) == n_seats - 1
        self.env = env
        self.other_players = other_players
        self.n_seats = n_seats
        self._debug = debug
        self._encoder = encoder if not encoder is None else Encoder(n_seats, ranking_encoding=encoding)
        self._decoder = decoder if not decoder is None else Decoder()
        self._add_players(n_seats)

    @staticmethod
    def build_environment(opponent, n_seats, encoding='norm', equity_steps=100, debug=False):
        env = gym.make('TexasHoldem-v1', n_seats=n_seats, max_limit=MAX_BET, all_in_equity_reward=True, equity_steps=equity_steps, debug=debug)
        other_players = [opponent for i in range(n_seats - 1)]
        return TrainingEnv(env, other_players, n_seats, encoding=encoding, debug=debug)

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

    def swap_opponent_agent(self, new_agent):
        for opponent in self.other_players:
            opponent.swap_model(new_agent)

    def swap_other_players(self, other_players):
        self.other_players = other_players
    
    def reset(self):
        for p in self.env._seats:
            p.reset_stack()
        (player_states, (community_infos, community_cards)) = self.env.reset()
        self.start_stacks = []
        players, _ = zip(*player_states)
        for player in players:
            self.start_stacks.append(player[player_table.STACK])
        self.our_seat = players[0][player_table.SEAT_ID]
        if self._debug:
            print('Starting new round:', self.env._street)
            print("Letting others play after reset...")
        player_states, community_infos, community_cards, reward, done, info =\
            self._other_players_play(player_states, community_infos, community_cards)
        self.done_on_reset = done
        if done:
            self.reset_returns = (player_states, community_infos, community_cards, reward, done, info)
        encoded_state = self._encoder.encode(player_states, community_infos, community_cards, 0)
        self.outer_start_time = time.time()
        return encoded_state
    
    def step(self, action):
        start = time.time()
        outer_duration = time.time() - self.outer_start_time
        our_id = 0
        if self.done_on_reset:
            player_states, community_infos, community_cards, reward, done, info = self.reset_returns

        if self.done_on_reset:
            if self._debug:
                print('Opponent folded, starting new round...')
            player_states, community_infos, community_cards, reward, done, info = self.reset_returns
        else:
            move = self._decoder.decode(action, self.n_seats, self.minimum_raise, self.maximum_raise, 
                                        self.pot_size, self.amount_to_call, debug=self._debug)
            actions = np.zeros((self.n_seats, 2))
            actions[our_id,:] = move
            if self._debug:
                print('Action from Training Agent', actions)
                print('Round:', self.env._street)
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
        while to_act_pos != self.our_seat and self.env._street < 5 and not done: 
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
                print('Street:', self.env._street)
            states, reward, done, info = self.env.step(actions.astype(dtype=int))
            player_states, (community_infos, community_cards) = states

            if self._debug:
                self.render()
            to_act_pos = community_infos[community_table.TO_ACT_POS]

        return player_states, community_infos, community_cards, reward, done, info

    def _add_players(self, n_players):
        for i in range(n_players):
            self.env.add_player(i, stack=2000)
