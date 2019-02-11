from keras.callbacks import History

class PokerHistory(History):
    def on_step_end(self, step, logs=None):
        self.history.setdefault('money_won', []).append(logs['info']['money_won'])
        # self.history.setdefault('step_duration', []).append(logs['info']['step_duration'])
        # self.history.setdefault('outer_duration', []).append(logs['info']['outer_duration'])
