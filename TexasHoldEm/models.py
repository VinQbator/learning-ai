from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout

def test_model(window_length, n_observations, n_actions):
    model = Sequential(name='test-%s' % n_observations)
    model.add(Flatten(input_shape=(window_length, n_observations)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_actions, activation='softmax'))
    return model

def simple_model(window_length, n_observations, n_actions):
    model = Sequential(name='simple-%s' % (n_observations *window_length))
    model.add(Flatten(input_shape=(window_length, n_observations)))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_actions, activation='softmax'))
    return model

def complex_model(window_length, n_observations, n_actions):
    model = Sequential(name='complex-%s' % (n_observations *window_length))
    model.add(Flatten(input_shape=(window_length, n_observations)))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(n_actions, activation='softmax'))
    return model