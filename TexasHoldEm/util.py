import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import gc

import tensorflow as tf
 
import keras.backend as K
from keras.callbacks import History

using_jupyter = False

def set_on_demand_memory_allocation(usage_cap=1.0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = usage_cap
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    K.set_session(sess)

def release_memory(agents):
    #sess = K.tensorflow_backend.get_session() # works on cpu without this
    try:
        K.tensorflow_backend.clear_session()
    except:
        print('clear session')
    try:
        for agent in agents:
            if not agent is None:
                del agent
    except:
        print('something something')
    try:
        pass
        #sess.close() # Works on cpu without this
    except:
        print('sess close')
    print('%s objects released from memory' % gc.collect())
    set_on_demand_memory_allocation()

def use_jupyter():
    global using_jupyter
    using_jupyter = True

def print_stats(history):
    assert isinstance(history, History)
    winnings = np.array(history.history['money_won'])
    hands = len(winnings[winnings != 0])
    total_winnings = winnings.sum()
    print('Total $ won:', total_winnings)
    print('Winrate BB/100:', total_winnings / hands * 4)
    print('Total hands:', hands)

def visualize_history(history):
    assert isinstance(history, History)
    episode_reward = np.array(history.history['episode_reward'])
    episode_reward = episode_reward[episode_reward != 0]
    #print(len(episode_reward), episode_reward)
    winnings = np.array(history.history['money_won'])
    winnings = winnings[winnings != 0]
    hands = len(episode_reward)
    avg_over = hands // 10
    def SMA(arr):
        N = len(arr)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = arr[max(0, t-avg_over):(t+1)].mean()
        return running_avg
    
    fig, axs = plt.subplots(1, 3, figsize=(20,7))
    axs[0].plot(SMA(episode_reward)[avg_over//2:], label=('reward'))
    axs[0].set_title("Episode Rewards")
    axs[0].legend()
    axs[1].plot(SMA(winnings)[avg_over//2:] * 4, label=('bb/100 SMA %s' % avg_over))
    axs[1].set_title("Winrate")
    axs[1].legend()
    axs[2].plot(np.cumsum(winnings) / 25, label=('BB Won'))
    axs[2].set_title("Winnings")
    axs[2].legend()
    if not using_jupyter:
        fig.show()
    print_stats(history)

def get_latest_iteration_name(pattern):
    files = os.listdir('./weights/')
    found = []  
    for entry in files:  
        if fnmatch.fnmatch(entry, pattern):
            found.append(entry)
    return sorted(found)[-1], len(found)