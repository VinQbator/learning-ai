import warnings
import gc

import keras.backend as K
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

from helpers.poker_history import PokerHistory
from util import visualize_history, print_stats, get_latest_iteration_name, release_memory
from players.ai_player import AIPlayer

# Suppress FutureWarnings that trash the output
warnings.simplefilter(action='ignore', category=FutureWarning)

def build_dqn_agent(model, n_actions, window_length, target_model_update=1e-3, optimizer=None, 
                enable_double_dqn=False, enable_dueling_network=True, dueling_type='avg',
                train_interval=100, policy=None, memory=None, n_warmup_steps=50, batch_size=32,
                gamma=.99, memory_interval=1):
    
    optimizer = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True) if optimizer is None else optimizer
    memory = SequentialMemory(limit=20000//window_length, window_length=window_length) if memory is None else memory
    policy = BoltzmannQPolicy() if policy is None else policy
    agent = DQNAgent(
        model=model, nb_actions=n_actions, memory=memory, nb_steps_warmup=n_warmup_steps, target_model_update=1e-3, policy=policy, 
        train_interval=train_interval, memory_interval=memory_interval, enable_dueling_network=enable_dueling_network, 
        enable_double_dqn=enable_double_dqn, gamma=gamma, batch_size=batch_size, dueling_type=dueling_type)
    agent.compile(optimizer, metrics=['mae'])
    return agent

def fit_agent(agent, env, n_steps, debug, history=None, start_from_scratch=False):
    if not start_from_scratch:
        load_agent_weights(agent)
    history = PokerHistory() if history is None else history
    history = agent.fit(env, nb_steps=n_steps, visualize=debug, log_interval=n_steps//5, verbose=1, history=history)
    return agent, history

# A method to iteratively keep playing against previous versions of ourselves
def train_loop(agent, model, env, steps_in_iteration, n_iterations, window_length, history=None, debug=False):
    env.swap_other_players([AIPlayer(agent) for i in range(len(env.other_players))])
    history = PokerHistory() if history is None else history
    opponent_agent = None
    save_agent_weights(agent)
    for i in range(n_iterations):
        print('ITERATION %s' % str(i))
        # Free up resources first
        release_memory([opponent_agent, agent])
        # Create a copy of the agent to play against us (and to free up the resources recreate our training agent as well)
        agent = build_dqn_agent(model(window_length, env.n_observation_dimensions, env.n_actions), env.n_actions, window_length, debug)
        #load_agent_weights(agent)
        opponent_agent = build_dqn_agent(model(window_length, env.n_observation_dimensions, env.n_actions), env.n_actions, window_length, debug)
        #load_agent_weights(opponent_agent)
        env.swap_opponent_agent(opponent_agent)
        history = agent.fit(env, nb_steps=steps_in_iteration, visualize=debug, log_interval=steps_in_iteration//5, verbose=0, history=history)
        print_stats(history)
        save_agent_weights(agent)
    
    release_memory([opponent_agent, agent])
    # Create a copy of the agent to play against us (and to free up the resources recreate our training agent as well)
    agent = build_dqn_agent(model(window_length, env.n_observation_dimensions, env.n_actions), env.n_actions, window_length, debug)
    #load_agent_weights(agent)
    opponent_agent = build_dqn_agent(model(window_length, env.n_observation_dimensions, env.n_actions), env.n_actions, window_length, debug)
    #load_agent_weights(opponent_agent)
    env.swap_opponent_agent(opponent_agent)
    return agent, history

def agent_weight_name(agent):
    return 'loop-%s' % agent.model.name

def load_agent_weights(agent):
    try:
        agent.load_weights('weights/' + agent_weight_name(agent))
    except:
        print('Could not load previous weights')

def save_agent_weights(agent, overwrite=True):

    # TODO: save safely (buffer to temp file and on successful save overwrite original)

    agent.save_weights('weights/' + agent_weight_name(agent), overwrite=overwrite)