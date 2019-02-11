import numpy as np
import matplotlib.pyplot as plt

from keras.callbacks import History

def visualize_history(history):
    assert isinstance(history, History)
    avg_over = len(history.history['episode_reward']) // 10
    def SMA(arr):
        N = len(arr)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = arr[max(0, t-avg_over):(t+1)].mean()
        return running_avg
    
    episode_reward = np.array(history.history['episode_reward'])
    winnings = np.array(history.history['money_won'])
    winnings = winnings[np.nonzero(winnings)]
    plt.plot(SMA(episode_reward)[avg_over//2:], label=('reward'))
    plt.title("Episode Rewards")
    plt.legend()
    plt.show()
    plt.plot(SMA(winnings)[avg_over//2:] *4, label=('bb/100 SMA %s' % avg_over))
    plt.title("Winrate")
    plt.legend()
    plt.show()
    plt.plot(np.cumsum(winnings - (10 + 25) / 2) / 25, label=('BB Won'))
    plt.title("Winnings")
    plt.legend()
    plt.show()