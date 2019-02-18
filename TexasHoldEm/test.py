import time

import gym
from holdem.env import Env

from training_env import TrainingEnv
from players.atm import ATM
from encoders import Encoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

N_PLAYERS = 10

def main():
    env = gym.make('TexasHoldem-v1', n_seats=N_PLAYERS, max_limit=100000, all_in_equity_reward=True, equity_steps=1000, debug=False)
    for i in range(N_PLAYERS):
        env.add_player(i, stack=2000)
    for p in env._seats:
        p.reset_stack()
    state = env.reset()
    (player_states, (community_infos, community_cards)) = state
    encoder = Encoder(N_PLAYERS, ranking_encoding='norm')
    start_time = time.time()
    encoded_state = encoder.encode(player_states, community_infos, community_cards, 0)
    time_taken = time.time() - start_time
    print(time_taken)
    print(encoded_state.shape)
    tr_env = TrainingEnv.build_environment('asd', N_PLAYERS)
    print(tr_env.n_observation_dimensions)

if __name__ == '__main__':
    main()