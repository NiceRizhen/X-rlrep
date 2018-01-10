import gym
import numpy as np
import tensorflow as tf
from algos.base import RLAlgorithm
from algos.agent import Agent
from algos.train_agent import trainAgent
import matplotlib.pyplot as plt

METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPOAgent(Agent):

    def __init__(
            self,
            env,
            ep_max,
            ep_len,
            gamma,
            a_lr,
            c_lr,
            batch,
            a_update_steps,
            c_update_steps,
            s_dim,
            a_dim,
            ):
        self.env=env
        self.ep_max=ep_max
        self.ep_len=ep_len
        self.gamma=gamma
        self.a_lr=a_lr
        self.c_lr=c_lr
        self.batch=batch
        self.a_update_steps=a_update_steps
        self.c_update_steps=c_update_steps
        self.s_dim=s_dim
        self.a_dim=a_dim
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, self.s_dim], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def observe(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(self.a_update_steps):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.a_update_steps)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.c_update_steps)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, self.a_dim, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, self.a_dim, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def trainPolicy(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def runPolicy(self, s):
        pass

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class PPO(RLAlgorithm):
    def __init__(
            self,
            env,
            ep_max,
            ep_len,
            gamma,
            a_lr,
            c_lr,
            batch,
            a_update_steps,
            c_update_steps,
            s_dim,
            a_dim,
            ):
        self.env=env
        self.ep_max=ep_max
        self.ep_len=ep_len
        self.gamma=gamma
        self.a_lr=a_lr
        self.c_lr=c_lr
        self.batch=batch
        self.a_update_steps=a_update_steps
        self.c_update_steps=c_update_steps
        self.s_dim=s_dim
        self.a_dim=a_dim

    def train(self):
        self.env = self.env.unwrapped

        ppo = PPOAgent(
            env=self.env,
            ep_max=self.ep_max,
            ep_len=self.ep_len,
            gamma=self.gamma,
            a_lr=self.a_lr,
            c_lr=self.c_lr,
            batch=self.batch,
            a_update_steps=self.a_update_steps,
            c_update_steps=self.c_update_steps,
            s_dim=self.s_dim,
            a_dim=self.a_dim
        )
        all_ep_r = []
        history = []

        for ep in range(self.ep_max):
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            for t in range(self.ep_len):    # in one episode
                # self.env.render()
                a = ppo.trainPolicy(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r+8)/8)    # normalize reward, find to be useful
                s = s_
                ep_r += r

                # update ppo
                if (t+1) % self.batch == 0 or t == self.ep_len-1:
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + self.gamma * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.observe(bs, ba, br)

            history.append(ep_r)
            if ep == 0: all_ep_r.append(ep_r)
            else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
                ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
            )

        plt.figure(1)
        plt.plot(np.array(history), c='b', label='PPO')
        plt.legend(loc='best')
        plt.ylabel('reward')
        plt.xlabel('episode')
        plt.grid()

        plt.show()
