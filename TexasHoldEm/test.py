import time

import gym
from holdem.env import Env

from training_env import TrainingEnv
from players.atm import ATM
from encoders import Encoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

N_PLAYERS = 2

def main():
    env = gym.make('TexasHoldem-v1', n_seats=N_PLAYERS, max_limit=100000, all_in_equity_reward=True, equity_steps=1000, debug=False)
    for i in range(N_PLAYERS):
        env.add_player(i, stack=2000)
    for p in env._seats:
        p.reset_stack()
    state = env.reset()
    #print(state)
    (player_states, (community_infos, community_cards)) = state
    #print('player states', player_states)
    #print('community info', community_infos)
    encoder = Encoder(N_PLAYERS, ranking_encoding=None)
    start_time = time.time()
    #community_info, players_info, community_cards, player_cards = encoder.encode(player_states, community_infos, community_cards, 0, concat=False)
    encoded_state = encoder.encode(player_states, community_infos, community_cards, 0)
    time_taken = time.time() - start_time
    print(time_taken)
    #print(encoded_state.shape)
    tr_env = TrainingEnv.build_environment('asd', N_PLAYERS)
    #print(tr_env.n_observation_dimensions)
    #print('Community Info:', community_info)
    #print('Players Info:', players_info)
    #print('Community Cards:', community_cards)
    #print('Player Cards:', player_cards)
    #print('Hand Rank:', hand_rank)
    #print(encoder.encode_slow(player_states, community_infos, community_cards, 0))
    step = [[1, 0]] * N_PLAYERS
    #print(step)
    state, reward, done, info = env.step(step)
    (player_states, (community_infos, community_cards)) = state
    encoded_state = encoder.encode(player_states, community_infos, community_cards, 0)

    step = [[0, 0]] * N_PLAYERS
    state, reward, done, info = env.step(step)
    (player_states, (community_infos, community_cards)) = state
    encoded_state = encoder.encode(player_states, community_infos, community_cards, 0)


if __name__ == '__main__':
    main()