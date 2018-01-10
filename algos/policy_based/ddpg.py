import numpy as np
import tensorflow as tf
from algos.base import RLAlgorithm
from algos.agent import Agent
import matplotlib.pyplot as plt


class DDPGAgent(Agent):
    def __init__(
            self,
            env,
            a_dim,
            s_dim,
            a_bound,
            max_episodes,
            max_ep_steps,
            lr_a,
            lr_c,
            gamma,
            tau,
            memory_capacity,
            batch_size
            ):
        self.env=env
        self.a_dim=a_dim
        self.s_dim=s_dim
        self.a_bound=a_bound
        self.max_episodes=max_episodes
        self.max_ep_steps=max_ep_steps
        self.lr_a=lr_a
        self.lr_c=lr_c
        self.gamma=gamma
        self.tau=tau
        self.memory_capacity=memory_capacity
        self.batch_size=batch_size
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.tau) * ta + self.tau * ea), tf.assign(tc, (1 - self.tau) * tc + self.tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.gamma * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def trainPolicy(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def runPolicy(self, s):
        pass

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def observe(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


class DDPG(RLAlgorithm):
    def __init__(
            self,
            env,
            a_dim,
            s_dim,
            a_bound,
            max_episodes,
            max_ep_steps,
            lr_a,
            lr_c,
            gamma,
            tau,
            memory_capacity,
            batch_size
    ):
        self.env=env
        self.a_dim=a_dim
        self.s_dim=s_dim
        self.a_bound=a_bound
        self.max_episodes=max_episodes
        self.max_ep_steps=max_ep_steps
        self.lr_a=lr_a
        self.lr_c=lr_c
        self.gamma=gamma
        self.tau=tau
        self.memory_capacity=memory_capacity
        self.batch_size=batch_size

    def train(self):
        ddpg = DDPGAgent(
                env=self.env,
                a_dim=self.a_dim,
                s_dim=self.s_dim,
                a_bound=self.a_bound,
                max_episodes=self.max_episodes,
                max_ep_steps=self.max_ep_steps,
                lr_a=self.lr_a,
                lr_c=self.lr_c,
                gamma=self.gamma,
                tau=self.tau,
                memory_capacity=self.memory_capacity,
                batch_size=self.batch_size
                )
        history = []
        var = 3  # control exploration
        for i in range(self.max_episodes):
            s = self.env.reset()
            ep_reward = 0
            for j in range(self.max_ep_steps):

                # Add exploration noise
                a = ddpg.trainPolicy(s)
                a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
                s_, r, done, info = self.env.step(a)

                ddpg.observe(s, a, r / 10, s_)

                if ddpg.pointer > self.memory_capacity:
                    var *= .9995    # decay the action randomness
                    ddpg.learn()

                s = s_
                ep_reward += r
                if j == self.max_ep_steps-1:
                    print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                    # if ep_reward > -300:RENDER = True
                    break
            history.append(ep_reward)

        plt.figure(1)
        plt.plot(np.array(history), c='b', label='DDPG')
        plt.legend(loc='best')
        plt.ylabel('reward')
        plt.xlabel('episode')
        plt.grid()

        plt.show()
