#!/usr/bin/env python3
"""
rl.py

Enhanced DQNAgent:
 - Optionally uses Double DQN
 - Optionally uses Huber loss
 - Includes L2 regularization in Dense layers
 - Discrete actions {LONG, SHORT, HOLD}
"""

import logging
import random
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.losses import Huber

from .environment import (
    RL_MODEL_PATH,
    NUM_FUTURE_STEPS
)

ACTIONS = ["LONG", "SHORT", "HOLD"]

class DQNAgent:
    def __init__(
        self,
        state_dim=NUM_FUTURE_STEPS+3,
        gamma=0.99,
        lr=0.001,
        batch_size=32,
        max_memory=20000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        update_target_steps=1000,
        double_dqn=True,
        loss_fn="huber_loss",
        l2_reg=1e-6
    ):
        """
        :param state_dim: dimension of the RL state
        :param gamma: discount factor
        :param lr: learning rate
        :param batch_size: mini-batch size
        :param max_memory: replay buffer capacity
        :param epsilon_start: initial epsilon for exploration
        :param epsilon_min: minimum epsilon
        :param epsilon_decay: exponential decay per step
        :param update_target_steps: how often to update the target_net
        :param double_dqn: whether to use Double DQN
        :param loss_fn: 'mse' or 'huber_loss'
        :param l2_reg: L2 regularization factor
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
        self.double_dqn = double_dqn
        self.loss_fn = loss_fn
        self.l2_reg = l2_reg

        self.memory = deque(maxlen=max_memory)
        self.learn_step_counter = 0

        self.q_net = self._build_network(lr)
        self.target_net = self._build_network(lr)
        self._update_target()

        # Attempt to load existing weights
        try:
            self.load()
            self.logger.info("Loaded DQNAgent weights from " + RL_MODEL_PATH)
        except FileNotFoundError:
            self.logger.warning(RL_MODEL_PATH + " not found.")
        except OSError:
            self.logger.warning(RL_MODEL_PATH + " not found.")

    def _build_network(self, lr):
        """
        Simple 2-layer feed-forward network with optional L2 reg.
        We allow switching between Huber and MSE.
        """
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(self.state_dim,)))
        model.add(layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))
        model.add(layers.Dense(
            64, activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))
        model.add(layers.Dense(
            self.action_dim,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))

        if self.loss_fn == "huber_loss":
            loss = Huber()
        else:
            loss = "mse"

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=loss
        )
        return model

    def _update_target(self):
        self.target_net.set_weights(self.q_net.get_weights())

    def select_action(self, state_vec):
        """
        Epsilon-greedy action selection.
        :param state_vec: np.array of shape (state_dim,)
        """
        if np.random.rand() < self.epsilon:
            a_idx = np.random.choice(self.action_dim)
        else:
            q_values = self.q_net.predict(state_vec[np.newaxis,:], verbose=0)
            a_idx = np.argmax(q_values[0])
        return ACTIONS[a_idx]

    def store_transition(self, state, action, reward, next_state, done):
        a_idx = ACTIONS.index(action)
        self.memory.append((state, a_idx, reward, next_state, done))

    def train_step(self):
        """
        Sample from replay, do one gradient update.
        Uses Double DQN or regular DQN depending on self.double_dqn.
        """
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for (s, a, r, s2, d) in batch:
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

        # Current Q-values
        q_vals = self.q_net.predict(states, verbose=0)

        # Next Q-values (from target)
        q_next_target = self.target_net.predict(next_states, verbose=0)

        if self.double_dqn:
            # Double DQN uses online net to select action (argmax),
            # but target net to evaluate that action's Q-value.
            q_next_online = self.q_net.predict(next_states, verbose=0)
            next_actions = np.argmax(q_next_online, axis=1)

        for i in range(self.batch_size):
            a = actions[i]
            if dones[i]:
                # Terminal state
                q_vals[i][a] = rewards[i]
            else:
                if self.double_dqn:
                    # Double DQN
                    q_vals[i][a] = rewards[i] + self.gamma * q_next_target[i][ next_actions[i] ]
                else:
                    # Regular DQN
                    q_vals[i][a] = rewards[i] + self.gamma * np.max(q_next_target[i])

        # One gradient descent step
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
        Load model weights from RL_MODEL_PATH.
        """
        self.q_net.load_weights(RL_MODEL_PATH)
        self.target_net.load_weights(RL_MODEL_PATH)
        self.logger.info(f"Loaded DQN weights from {RL_MODEL_PATH}")

    def save(self):
        """
        Save model weights to RL_MODEL_PATH.
        """
        self.q_net.save_weights(RL_MODEL_PATH)
        self.logger.info(f"DQN weights saved to {RL_MODEL_PATH}")
