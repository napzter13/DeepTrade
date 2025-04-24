"""
Reinforcement Learning Agent for Trading

This is an enhanced implementation with:
- Double DQN for more stable learning
- Prioritized experience replay for better sample utilization
- Dueling network architecture for better value estimation
- Memory optimization for reduced resource usage
"""

import os
import numpy as np
import tensorflow as tf
import random
import json
from pathlib import Path
import logging

# Initialize logging
logger = logging.getLogger("RL")

# Actions available to the agent
ACTIONS = ["LONG", "SHORT", "HOLD"]

class DQNAgent:
    """
    DQN Agent with Double DQN, Dueling Networks, and Prioritized Experience Replay
    """
    def __init__(
        self,
        state_dim=13,
        action_space=ACTIONS,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.995,
        lr=0.001,
        batch_size=64,
        max_memory=10000,
        update_target_steps=100,
        use_double_dqn=True,
        use_dueling=True,
        use_prioritized_replay=True,
        alpha=0.6,  # prioritization exponent
        beta_start=0.4,  # importance sampling exponent
        beta_increment=0.001
    ):
        """Initialize the DQN agent with enhanced learning algorithms"""
        self.state_dim = state_dim
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = []
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.batch_size = batch_size
        self.max_memory = max_memory
        self.update_target_steps = update_target_steps
        self.use_double_dqn = use_double_dqn
        self.use_dueling = use_dueling
        self.use_prioritized_replay = use_prioritized_replay
        
        # For prioritized replay
        if use_prioritized_replay:
            self.priorities = np.zeros(max_memory, dtype=np.float32)
            self.alpha = alpha  # prioritization exponent
            self.beta = beta_start  # importance sampling exponent
            self.beta_increment = beta_increment
            self.epsilon_prio = 1e-6  # small constant to avoid zero priority
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_update_counter = 0
        self.update_target_model()
        
        # Metrics for monitoring
        self.loss_history = []
        
        logger.info(f"DQN agent initialized with state_dim={state_dim}, actions={action_space}")
        logger.info(f"Using double_dqn={use_double_dqn}, dueling={use_dueling}, prioritized_replay={use_prioritized_replay}")
        
    def _build_model(self):
        """Build DQN model with dueling architecture option"""
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        
        # Initial layers
        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        if self.use_dueling:
            # Value stream
            value_stream = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
            value_stream = tf.keras.layers.BatchNormalization()(value_stream)
            value = tf.keras.layers.Dense(1)(value_stream)
            
            # Advantage stream
            advantage_stream = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform')(x)
            advantage_stream = tf.keras.layers.BatchNormalization()(advantage_stream)
            advantage = tf.keras.layers.Dense(self.action_size)(advantage_stream)
            
            # Combine streams
            # Q(s,a) = V(s) + A(s,a) - mean(A(s))
            outputs = tf.keras.layers.Add()([
                value,
                tf.keras.layers.Subtract()([
                    advantage,
                    tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(advantage)
                ])
            ])
        else:
            # Standard Q-network
            outputs = tf.keras.layers.Dense(self.action_size, kernel_initializer='he_uniform')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=1.0),
            loss='mse'
        )
        return model
    
    def update_target_model(self):
        """Update target model weights from main model"""
        self.target_model.set_weights(self.model.get_weights())
        logger.debug("Target model weights updated")
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        # Convert string action to index
        if isinstance(action, str):
            action_idx = self.action_space.index(action)
        else:
            action_idx = action
            
        # Create memory entry
        memory_item = (state, action_idx, reward, next_state, done)
        
        # Add to memory
        if len(self.memory) < self.max_memory:
            self.memory.append(memory_item)
        else:
            # Replace old experiences
            idx = len(self.memory) % self.max_memory
            self.memory[idx] = memory_item
            
        # For prioritized replay
        if self.use_prioritized_replay:
            # New experiences get max priority
            max_priority = max(np.max(self.priorities), 1.0)
            
            if len(self.memory) < self.max_memory:
                self.priorities = np.append(self.priorities, max_priority)
            else:
                idx = len(self.memory) % self.max_memory
                self.priorities[idx] = max_priority
    
    def select_action(self, state):
        """Choose action based on state using epsilon-greedy policy"""
        # Exploration
        if np.random.rand() < self.epsilon:
            return random.choice(self.action_space)
        
        # Exploitation
        state_tensor = np.array(state).reshape(1, -1).astype(np.float32)
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        action_idx = np.argmax(q_values)
        
        return self.action_space[action_idx]
    
    def train_step(self):
        """Train the agent by sampling from replay memory"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Choose appropriate replay method
        if self.use_prioritized_replay:
            return self._prioritized_replay()
        else:
            return self._uniform_replay()
    
    def _uniform_replay(self):
        """Standard uniform experience replay"""
        # Sample random minibatch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Initialize batch arrays
        states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.int32)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        dones = np.zeros(self.batch_size, dtype=np.bool_)
        
        # Fill batch arrays
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Compute targets differently depending on algorithm
        if self.use_double_dqn:
            # Double DQN: Use main network to select actions, target network to evaluate
            next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
            next_q = self.target_model.predict(next_states, verbose=0)
            targets = rewards + self.gamma * next_q[np.arange(self.batch_size), next_actions] * (1 - dones)
        else:
            # Standard DQN
            next_q = self.target_model.predict(next_states, verbose=0)
            targets = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)
        
        # Update Q values for taken actions
        for i in range(self.batch_size):
            current_q[i, actions[i]] = targets[i]
        
        # Train the model
        history = self.model.fit(states, current_q, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update target model periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_steps == 0:
            self.update_target_model()
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def _prioritized_replay(self):
        """Prioritized experience replay"""
        # Calculate sampling probabilities
        memory_size = len(self.memory)
        priorities = self.priorities[:memory_size]
        probs = priorities ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample indices based on priorities
        indices = np.random.choice(memory_size, self.batch_size, p=probs)
        
        # Initialize batch arrays
        states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.int32)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        next_states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        dones = np.zeros(self.batch_size, dtype=np.bool_)
        
        # Fill batch arrays
        for i, idx in enumerate(indices):
            state, action, reward, next_state, done = self.memory[idx]
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done
        
        # Calculate importance sampling weights
        # w_i = (N * P(i))^(-beta)
        weights = (memory_size * probs[indices]) ** -self.beta
        weights = weights / np.max(weights)  # Normalize for stability
        
        # Get current Q values
        current_q = self.model.predict(states, verbose=0)
        
        # Compute targets (as in Double DQN / standard DQN)
        if self.use_double_dqn:
            next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)
            next_q = self.target_model.predict(next_states, verbose=0)
            targets = rewards + self.gamma * next_q[np.arange(self.batch_size), next_actions] * (1 - dones)
        else:
            next_q = self.target_model.predict(next_states, verbose=0)
            targets = rewards + self.gamma * np.max(next_q, axis=1) * (1 - dones)
        
        # Calculate TD errors for updating priorities
        td_errors = np.zeros(self.batch_size)
        
        # Update Q values for taken actions
        for i in range(self.batch_size):
            td_errors[i] = abs(targets[i] - current_q[i, actions[i]])
            current_q[i, actions[i]] = targets[i]
        
        # Train model with importance sampling weights
        history = self.model.fit(states, current_q, sample_weight=weights, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        
        # Update priorities
        for i, idx in enumerate(indices):
            self.priorities[idx] = td_errors[i] + self.epsilon_prio
        
        # Increase beta over time (annealing)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Update target model periodically
        self.target_update_counter += 1
        if self.target_update_counter % self.update_target_steps == 0:
            self.update_target_model()
            
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss
    
    def save(self, filepath="models/rl_DQNAgent.weights.h5"):
        """Save model weights"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            self.model.save_weights(filepath)
            logger.info(f"Model weights saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load(self, filepath="models/rl_DQNAgent.weights.h5"):
        """Load model weights"""
        if not os.path.exists(filepath):
            logger.warning(f"Model weights file not found: {filepath}")
            return False
            
        try:
            self.model.load_weights(filepath)
            self.update_target_model()
            logger.info(f"Model weights loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False