#!/usr/bin/env python3
"""
rl.py

Contains two RL classes:

DQNAgent (a new Deep Q-Network-based agent for advanced RL).
   - It maintains a replay buffer, trains a small feed-forward net to
     approximate Q-values for discrete actions {LONG, SHORT, HOLD}.
"""

import logging
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers

from .environment import (
    RL_MODEL_PATH
)


###############################################################################
# DQNAgent (Deep Q-Network)
###############################################################################
ACTIONS = ["LONG", "SHORT", "HOLD"]

class DQNAgent:
    def __init__(
        self,
        state_dim=6,         # dimension of the RL state vector
        gamma=0.99,
        lr=0.001,
        batch_size=32,
        max_memory=20000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        update_target_steps=1000
    ):
        """
        :param state_dim: dimension of the RL state we feed to the DQN
        :param gamma: discount factor
        :param lr: learning rate for Adam
        :param batch_size: mini-batch size for training
        :param max_memory: replay buffer capacity
        :param epsilon_start: initial exploration rate
        :param epsilon_min: minimum exploration
        :param epsilon_decay: exponential decay per step
        :param update_target_steps: freq to update target net
        """
        self.logger = logging.getLogger("DQNAgent")
        self.logger.setLevel(logging.INFO)

        self.state_dim = state_dim
        self.action_dim = len(ACTIONS)

        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.update_target_steps = update_target_steps

        # Replay buffer
        self.memory = deque(maxlen=max_memory)
        self.learn_step_counter = 0

        # Build Q-network and target network
        self.q_net = self._build_network(lr)
        self.target_net = self._build_network(lr)
        self._update_target()
        
        try:
            self.load()
            self.logger.info("Loaded DQNAgent weights from " + RL_MODEL_PATH)
        except FileNotFoundError:
            self.logger.warning(RL_MODEL_PATH + " not found.")

    def _build_network(self, lr):
        """
        A small feed-forward network mapping state_dim -> Q-values for 3 actions.
        """
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_dim,)))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(self.action_dim, activation='linear'))
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss='mse'
        )
        return model

    def _update_target(self):
        """ Copy weights from q_net to target_net. """
        self.target_net.set_weights(self.q_net.get_weights())

    def select_action(self, state_vec):
        """
        Epsilon-greedy action selection.
        :param state_vec: np.array of shape (state_dim,)
        """
        if np.random.rand() < self.epsilon:
            # random
            action_idx = np.random.choice(self.action_dim)
        else:
            q_values = self.q_net.predict(state_vec[np.newaxis,:], verbose=0)
            action_idx = np.argmax(q_values[0])

        return ACTIONS[action_idx]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Save to replay buffer: (state_vec, action_idx, reward, next_state_vec, done)
        """
        action_idx = ACTIONS.index(action)
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_step(self):
        """
        Sample from replay buffer, do a single DQN update.
        """
        if len(self.memory) < self.batch_size:
            return  # not enough data to train

        mini_batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = [],[],[],[],[]

        for (s,a,r,s2,d) in mini_batch:
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s2)
            dones.append(d)

        states = np.array(states, dtype=np.float32)
        actions= np.array(actions, dtype=np.int32)
        rewards= np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones  = np.array(dones, dtype=bool)

        # Predict Q for current states
        q_vals = self.q_net.predict(states, verbose=0)
        # Predict Q' for next states (using target net)
        q_next = self.target_net.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            a = actions[i]
            if dones[i]:
                q_vals[i][a] = rewards[i]
            else:
                q_vals[i][a] = rewards[i] + self.gamma * np.max(q_next[i])

        # Train on updated Q values
        self.q_net.fit(states, q_vals, batch_size=self.batch_size, verbose=0)

        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self._update_target()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon, self.epsilon_min)

    def load(self):
        """
        Optionally load pretrained weights and set to both q_net & target_net
        """
        try:
            self.q_net.load_weights(RL_MODEL_PATH)
            self.target_net.load_weights(RL_MODEL_PATH)
            self.logger.info(f"Loaded DQN weights from {RL_MODEL_PATH}")
        except Exception as e:
            self.logger.warning(f"Could not load DQN weights: {e}")

    def save(self):
        """
        Save Q-network weights
        """
        self.q_net.save_weights(RL_MODEL_PATH)
        self.logger.info(f"DQN weights saved to {RL_MODEL_PATH}")
