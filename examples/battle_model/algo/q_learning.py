import os
import tensorflow as tf
import numpy as np

from . import base
from . import tools


class DQN(base.ValueNet):

    def __init__(self,
                 sess,
                 name,
                 handle,
                 env,
                 sub_len,
                 memory_size=2**10,
                 batch_size=64,
                 update_every=5):

        super().__init__(sess, env, handle, name, update_every=update_every)

        self.replay_buffer = tools.MemoryGroup(self.view_space,
                                               self.feature_space,
                                               self.num_actions, memory_size,
                                               batch_size, sub_len)
        self.sess.run(tf.global_variables_initializer())

    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        self.replay_buffer.tight()
        batch_num = self.replay_buffer.get_batch_num()

        for i in range(batch_num):
            obs, feats, obs_next, feat_next, dones, rewards, actions, masks = self.replay_buffer.sample(
            )
            target_q = self.calc_target_q(obs=obs_next,
                                          feature=feat_next,
                                          rewards=rewards,
                                          dones=dones)
            loss, q = super().train(state=[obs, feats],
                                    target_q=target_q,
                                    acts=actions,
                                    masks=masks)

            self.update()

            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    def save(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    def load(self, dir_path, step=0):
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       self.name_scope)
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "dqn_{}".format(step))

        saver.restore(self.sess, file_path)
        print("[*] Loaded model from {}".format(file_path))


class MFQ(base.ValueNet):

    def __init__(self,
                 sess,
                 name,
                 handle,
                 env,
                 sub_len,
                 eps=1.0,
                 update_every=5,
                 memory_size=2**10,
                 batch_size=64):
        # 调用父类的初始化方法
        super().__init__(sess,
                         env,
                         handle,
                         name,
                         use_mf=True,
                         update_every=update_every)

        config = {
            'max_len': memory_size,
            'batch_size': batch_size,
            'obs_shape': self.view_space,
            'feat_shape': self.feature_space,
            'act_n': self.num_actions,
            'use_mean': True,
            'sub_len': sub_len
        }

        self.train_ct = 0
        # 根据tools.py中的方法创建一个实例赋值为self的属性（用于存储和管理训练数据）
        self.replay_buffer = tools.MemoryGroup(**config)

        # 控制更新频率
        self.update_every = update_every

    # 为每个智能体存储数据（更新各个智能体的缓存区）
    def flush_buffer(self, **kwargs):
        self.replay_buffer.push(**kwargs)

    def train(self):
        # # 将各智能体的数据从各自的缓存区合并到主缓存区
        self.replay_buffer.tight()
        # 获取可以从回放缓冲区中采样的批次数量
        batch_name = self.replay_buffer.get_batch_num()

        # 遍历循环批次进行训练（batch）
        for i in range(batch_name):
            # 从回放缓冲区中采样数据，包括当前状态的观察值、特征值、动作、动作概率等
            obs, feat, acts, act_prob, obs_next, feat_next, act_prob_next, rewards, dones, masks = self.replay_buffer.sample(
            )
            # 调用calc_target_q方法计算目标 Q 值。
            target_q = self.calc_target_q(obs=obs_next,
                                          feature=feat_next,
                                          rewards=rewards,
                                          dones=dones,
                                          prob=act_prob_next)
            # 当子类重写了父类的方法时，还需要调用父类的方法，就需要使用super()而不是self来调用
            loss, q = super().train(state=[obs, feat],
                                    target_q=target_q,
                                    prob=act_prob,
                                    acts=acts,
                                    masks=masks)

            self.update()

            # 每50步打印损失和Q值信息
            if i % 50 == 0:
                print('[*] LOSS:', loss, '/ Q:', q)

    # 用于保存模型
    def save(self, dir_path, step=0):
        # 获取当前作用域下的全局变量集合
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       self.name_scope)
        # 创建一个保存器对象
        # 这个对象被配置为专门处理这些特定的变量，这一步是为了明确要保存哪些变量，使得Saver对象能够针对这些变量进行操作。
        saver = tf.train.Saver(model_vars)

        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        # 通过self.sess获取model_vars变量的实际值保存
        saver.save(self.sess, file_path)

        print("[*] Model saved at: {}".format(file_path))

    # 用于加载已保存的模型
    def load(self, dir_path, step=0):
        # 获取当前作用域下的全局变量集合（不是实际值，类似占位符的概念？）
        model_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                       self.name_scope)
        # 用全局变量创建保存器对象
        saver = tf.train.Saver(model_vars)
        file_path = os.path.join(dir_path, "mfq_{}".format(step))
        # 从指定文件路径加载模型变量到当前会话（sess）中
        saver.restore(self.sess, file_path)

        print("[*] Loaded model from {}".format(file_path))
