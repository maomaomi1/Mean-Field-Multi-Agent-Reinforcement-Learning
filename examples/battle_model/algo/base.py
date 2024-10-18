import tensorflow as tf
import numpy as np

from magent.gridworld import GridWorld


class ValueNet:

    def __init__(self,
                 sess,
                 env,
                 handle,
                 name,
                 update_every=5,
                 use_mf=False,
                 learning_rate=1e-4,
                 tau=0.005,
                 gamma=0.95):
        # assert isinstance(env, GridWorld)
        self.env = env
        self.name = name
        self._saver = None
        self.sess = sess

        # 识别特定智能体的句柄
        self.handle = handle
        # 环境获取的观测空间
        self.view_space = env.get_view_space(handle)
        assert len(self.view_space) == 3
        # 环境获取的特征空间
        self.feature_space = env.get_feature_space(handle)
        # 动作空间的大小（可选动作的数量）
        self.num_actions = env.get_action_space(handle)[0]

        # 控制更新频率
        self.update_every = update_every
        self.use_mf = use_mf  # trigger of using mean field
        # 用于控制策略（动作选择）的温度参数
        self.temperature = 0.1

        # 分别为学习率、软更新系数、折扣因子
        self.lr = learning_rate
        self.tau = tau
        self.gamma = gamma

        # 使用给定的name来创建一个变量作用域，如果name为空，就使用固定的"ValueNet"，并将这个作用域的name赋值给对象的属性
        with tf.variable_scope(name or "ValueNet"):
            self.name_scope = tf.get_variable_scope().name
            # 当我们在TensorFlow中构建模型时，我们需要定义哪些变量是模型的参数（通过tf.Variable），
            # 哪些变量是模型的输入（通过tf.placeholder）。这样，在训练或评估模型时，我们可以将实际的数据通过占位符传递给模型。
            # 也就是占位符的定义，先占位子
            self.obs_input = tf.placeholder(tf.float32,
                                            (None, ) + self.view_space,
                                            name="Obs-Input")
            self.feat_input = tf.placeholder(tf.float32,
                                             (None, ) + self.feature_space,
                                             name="Feat-Input")
            self.mask = tf.placeholder(tf.float32,
                                       shape=(None, ),
                                       name='Terminate-Mask')

            # 使用MF的话会有额外的占位符定义
            if self.use_mf:
                self.act_prob_input = tf.placeholder(tf.float32,
                                                     (None, self.num_actions),
                                                     name="Act-Prob-Input")

            # TODO: for calculating the Q-value, consider softmax usage
            self.act_input = tf.placeholder(tf.int32, (None, ), name="Act")
            # 将上面定义的act_input转换成独热编码
            self.act_one_hot = tf.one_hot(self.act_input,
                                          depth=self.num_actions,
                                          on_value=1.0,
                                          off_value=0.0)

            # 评估网络参数构建
            with tf.variable_scope("Eval-Net"):
                self.eval_name = tf.get_variable_scope().name
                # 调用自定义的函数来构建网络，输入为激活函数类型
                self.e_q = self._construct_net(active_func=tf.nn.relu)
                # 对评估网络的输出self.e_q进行softmax处理
                self.predict = tf.nn.softmax(self.e_q / self.temperature)
                # 获取在 “Eval-Net” 作用域下的所有全局变量
                self.e_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.eval_name)

            # 目标网络构建
            with tf.variable_scope("Target-Net"):
                self.target_name = tf.get_variable_scope().name
                self.t_q = self._construct_net(active_func=tf.nn.relu)
                self.t_variables = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=self.target_name)

            # 该作用域中定义了更新方式：t_variables（目标网络）和e_variables（评估网络）作用
            with tf.variable_scope("Update"):
                self.update_op = [
                    tf.assign(
                        self.t_variables[i], self.tau * self.e_variables[i] +
                        (1. - self.tau) * self.t_variables[i])
                    for i in range(len(self.t_variables))
                ]

            # 该作用域定义优化相关的操作
            with tf.variable_scope("Optimization"):
                # 用于接受目标q值
                self.target_q_input = tf.placeholder(tf.float32, (None, ),
                                                     name="Q-Input")
                # self.act_one_hot是动作的独热编码表示，self.e_q是评估网络的输出（Q 值估计）。这里通过矩阵乘法和在特定轴上求和的操作，
                # 计算出在给定动作下的最大评估 Q 值。具体来说，先将动作的独热编码与评估网络的输出相乘，然后在轴 1（通常对应于动作的维度）上求和，得到每个样本在当前动作下的最大 Q 值。
                self.e_q_max = tf.reduce_sum(tf.multiply(
                    self.act_one_hot, self.e_q),
                                             axis=1)
                # 定义损失函数：目标Q值与评估Q值之间的差的平方
                self.loss = tf.reduce_sum(
                    tf.square(self.target_q_input - self.e_q_max) *
                    self.mask) / tf.reduce_sum(self.mask)
                # 选择优化器：使用 Adam 优化器来最小化损失函数。self.lr是学习率
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                    self.loss)

    # 构建网络的函数
    # 该网络的作用：接收不同的输入，输出对应状态的动作价值（Q值）估计
    def _construct_net(self, active_func=None, reuse=False):
        # 创建一层网络"Conv1"，该层对输入self.obs_input进行二维卷积操作
        conv1 = tf.layers.conv2d(self.obs_input,
                                 filters=32,
                                 kernel_size=3,
                                 activation=active_func,
                                 name="Conv1")
        # 再对conv1进行卷积操作
        conv2 = tf.layers.conv2d(conv1,
                                 filters=32,
                                 kernel_size=3,
                                 activation=active_func,
                                 name="Conv2")
        # 对conv2的输出形状重塑，转换成一维向量
        flatten_obs = tf.reshape(
            conv2, [-1, np.prod([v.value for v in conv2.shape[1:]])])

        # dense为创建全连接层，units=256表示该层的输出维度为 256
        h_obs = tf.layers.dense(flatten_obs,
                                units=256,
                                activation=active_func,
                                name="Dense-Obs")

        # 对特征输入self.feat_input进行全连接层操作
        h_emb = tf.layers.dense(self.feat_input,
                                units=32,
                                activation=active_func,
                                name="Dense-Emb",
                                reuse=reuse)

        # 将观察数据的全连接层输出h_obs和特征输入的全连接层输出h_emb在第一维度上进行拼接，形成一个新的张量。
        concat_layer = tf.concat([h_obs, h_emb], axis=1)

        if self.use_mf:
            # 使用MF的话多定义几个全连接层
            # 1、动作概率输入过两层全连接
            prob_emb = tf.layers.dense(self.act_prob_input,
                                       units=64,
                                       activation=active_func,
                                       name='Prob-Emb')
            h_act_prob = tf.layers.dense(prob_emb,
                                         units=32,
                                         activation=active_func,
                                         name="Dense-Act-Prob")
            # 2、再与观察层拼接
            concat_layer = tf.concat([concat_layer, h_act_prob], axis=1)

        # 拼接好的张量再进行全连接层操作，过两层，输出64维
        dense2 = tf.layers.dense(concat_layer,
                                 units=128,
                                 activation=active_func,
                                 name="Dense2")
        out = tf.layers.dense(dense2,
                              units=64,
                              activation=active_func,
                              name="Dense-Out")

        # 最后全连接层输出，维度为num_actions
        q = tf.layers.dense(out, units=self.num_actions, name="Q-Value")

        return q

    # @property装饰器将函数定义为属性，这样在外部就可以直接obj.vars访问函数的输出，而不用加括号（）
    @property
    def vars(self):
        # 获取self.name_scope作用域下的全局变量集合
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                 scope=self.name_scope)

    def calc_target_q(self, **kwargs):
        # 用于计算目标Q值（期望Q值）
        """Calculate the target Q-value
        kwargs: {'obs', 'feature', 'prob', 'dones', 'rewards'}
        """
        # 其他参数：1、dones：布尔类型，表示当前状态是否为终止状态 2、rewards：当前状态下的奖励值
        feed_dict = {
            self.obs_input: kwargs['obs'],
            self.feat_input: kwargs['feature']
        }

        # 如果使用MF，就会引入act_prob_input即动作概率项
        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        # 计算目标Q值和评估Q值
        t_q, e_q = self.sess.run([self.t_q, self.e_q], feed_dict=feed_dict)
        # 根据评估Q值计算最优动作的索引
        act_idx = np.argmax(e_q, axis=1)
        # 提取最优动作在t_q表中的Q值，act_idx指定列，也就是说q_values为t_q的某一列，是一个一维向量
        q_values = t_q[np.arange(len(t_q)), act_idx]

        # 计算目标 Q 值。这里使用了强化学习中的贝尔曼方程的变体
        # self.gamma是折扣因子，用于控制未来奖励的重要性，它决定了模型对未来奖励的重视程度
        target_q_value = kwargs['rewards'] + (
            1. - kwargs['dones']) * q_values.reshape(-1) * self.gamma

        return target_q_value

    # 图计算 更新权重
    def update(self):
        """Q-learning update"""
        self.sess.run(self.update_op)

    # 选择动作actions
    def act(self, **kwargs):
        """Act
        kwargs: {'obs', 'feature', 'prob', 'eps'}
        """
        # 观察和特征输入
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1]
        }

        self.temperature = kwargs['eps']

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            assert len(kwargs['prob']) == len(kwargs['state'][0])
            feed_dict[self.act_prob_input] = kwargs['prob']

        # feed_dict是一个字典（dictionary），，它主要用于在运行计算图（computation graph）时为占位符（placeholders）提供具体的值
        # 这里是得到一个包含每个动作的概率的数组
        actions = self.sess.run(self.predict, feed_dict=feed_dict)
        # 返回数组中最大元素的索引（即返回概率最大动作的索引）
        actions = np.argmax(actions, axis=1).astype(np.int32)
        return actions

    def train(self, **kwargs):
        """Train the model
        kwargs: {'state': [obs, feature], 'target_q', 'prob', 'acts'}
        """
        # prob：动作概率；acts：实际执行的动作；masks：掩码，用于控制损失的计算范围等
        feed_dict = {
            self.obs_input: kwargs['state'][0],
            self.feat_input: kwargs['state'][1],
            self.target_q_input: kwargs['target_q'],
            self.mask: kwargs['masks']
        }

        if self.use_mf:
            assert kwargs.get('prob', None) is not None
            feed_dict[self.act_prob_input] = kwargs['prob']

        feed_dict[self.act_input] = kwargs['acts']

        # 应用当前的输入字典feed_dict跑网络
        # 因为跑三个网络所以对应三个输出[self.train_op, self.loss, self.e_q_max]
        _, loss, e_q = self.sess.run([self.train_op, self.loss, self.e_q_max],
                                     feed_dict=feed_dict)
        return loss, {
            'Eval-Q': np.round(np.mean(e_q), 6),
            'Target-Q': np.round(np.mean(kwargs['target_q']), 6)
        }
