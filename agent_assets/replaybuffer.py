import numpy as np
import agent_assets.A_hparameters as hp

class ReplayBuffer():
    """A on-policy replay buffer, no importance sampling
    """

    def __init__(self, observation_space, action_space):
        """
        Stores observation space in uint8 dtype
        """
        self.obs_space = observation_space
        self.action_space = action_space

        self.obs_names = list(observation_space.spaces)
        self.obs_buffer = {}
        for name in self.obs_names:
            self.obs_buffer[name] = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.N = hp.Buf.N

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, N):
        self._N = N
        self.discount_window = hp.Q_discount**np.arange(N)

    def store_step(self, observation:dict, action, reward, done) :
        """Give the original observation in uint8

        Parameters
        ----------
        observation : dict
            s_t
        action
            a_t
        reward
            r_t
        done
            d_t
        """
        for name, obs in observation.items() :
            self.obs_buffer[name].append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.done_buffer.append(done)

    def sample(self, need_next_obs=False):
        N = self.N
        batch_size = len(self.reward_buffer) - N

        # Cumulative N steps reward
        cum_rewards = []
        cum_dones = []
        for i in range(batch_size):
            # Always need 'current' reward, current done effects next reward
            done_mask = self.done_buffer[i:i+N-1]
            done_mask.insert(0, False)
            done_mask = np.cumsum(done_mask)
            cum_reward = self.discount_window * self.reward_buffer[i:i+N]\
                                              * np.logical_not(done_mask)
            cum_rewards.append(np.sum(cum_reward))

            cum_dones.append(np.any(self.done_buffer[i:i+N]))

        cum_rewards = np.array(cum_rewards, dtype=np.float32)
        cum_dones = np.array(cum_dones)

        obs = {}
        for name, buf in self.obs_buffer.items():
            obs[name] = np.array(buf[:batch_size], 
                                dtype=self.obs_space[name].dtype)
        nth_obs = {}
        for name, buf in self.obs_buffer.items():
            nth_obs[name] = np.array(buf[N:batch_size+N],
                                    dtype=self.obs_space[name].dtype)

        actions = np.array(self.action_buffer[:batch_size],
                            dtype=self.action_space.dtype)

        if need_next_obs:
            next_obs = {}
            for name, buf in self.obs_buffer.items():
                next_obs[name] = np.array(buf[1:batch_size+1],
                                        dtype=self.obs_space[name].dtype)
        else :
            next_obs = None

        return (obs, 
                actions, 
                cum_rewards, 
                cum_dones,
                nth_obs,
                next_obs)

    def reset_continue(self):
        """
        Leave last N unused samples
        """
        N = self.N
        for name, buf in self.obs_buffer.items():
            self.obs_buffer[name] = buf[-N:]
        self.action_buffer = self.action_buffer[-N:]
        self.reward_buffer = self.reward_buffer[-N:]
        self.done_buffer = self.done_buffer[-N:]

    def reset_all(self):
        """
        Clear all buffer
        """
        self.obs_buffer = {}
        for name in self.obs_names:
            self.obs_buffer[name] = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
