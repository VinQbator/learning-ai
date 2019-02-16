from keras.callbacks import History

class PokerHistory(History):
    def on_train_begin(self, logs=None):
        if not hasattr(self, 'epoch'):
            self.epoch = []
        if not hasattr(self, 'history'):
            self.history = {}

    def on_step_end(self, step, logs=None):
        self.history.setdefault('money_won', []).append(int(logs['info']['money_won']))
        self.history.setdefault('step_duration', []).append(logs['info']['step_duration'])
        self.history.setdefault('outer_duration', []).append(logs['info']['outer_duration'])
        self.history.setdefault('inner_duration', []).append(logs['info']['inner_duration'])
        self.history.setdefault('encode_duration', []).append(logs['info']['encode_duration'])
