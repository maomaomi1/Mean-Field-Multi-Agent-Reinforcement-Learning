import numpy as np
import tensorflow as tf
import os


# 定义常量，用于在终端中打印带有颜色的信息
class Color:
    INFO = '\033[1;34m{}\033[0m'
    WARNING = '\033[1;33m{}\033[0m'
    ERROR = '\033[1;31m{}\033[0m'


# 定义了一个抽象基类
class Buffer:

    def __init__(self):
        pass

    # 直接调用这个函数会抛出异常
    # 这种方式通常表明push方法需要在子类中实现
    def push(self, **kwargs):
        raise NotImplementedError


# 固定大小的环形缓冲区，可以存储多维数组数据
class MetaBuffer(object):

    def __init__(self, shape, max_len, dtype='float32'):

        # 缓冲区最大长度
        self.max_len = max_len

        # 初始化一个形状为 (max_len,) + shape 的零数组，用于存储数据
        # shape通常是一个元组，表示一个多维数组的形状，对于一个二维数组，shape可能是(m, n)，其中m是行数，n是列数
        # (max_len,) + shape) = (max_len, m, n)
        self.data = np.zeros((max_len, ) + shape).astype(dtype)

        # 起始索引
        self.start = 0
        # 当前缓冲区中数据的长度
        self.length = 0
        # 内部标志，用于管理环形缓冲区的位置，标志追加数据时的索引位置
        self._flag = 0

    def __len__(self):
        return self.length

    # 获取缓存区中指定索引的数据
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    # 也是获取指定索引的数据，通过取余，使得索引不会超出data长度，实现循环索引的效果
    # 可以有效应对循环或者周期性的数据访问
    # 实现环形缓存区
    def sample(self, idx):
        return self.data[idx % self.length]

    # 返回缓存区内所有有效数据
    def pull(self):
        return self.data[:self.length]

    # 向缓冲区中追加数据。如果追加的数据超过了缓冲区的最大长度，则会覆盖旧数据，实现环形缓冲区的功能。
    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    # 从指定的起始位置 start 开始，重置缓冲区中的数据
    def reset_new(self, start, value):
        self.data[start:] = value


# episode缓存区（一个训练回合：分多个时间步）
# 用于存储一个强化学习（或其他类似应用）中的单个 episode（即一次完整的交互序列）的数据
# 存储单个智能体的数据
class EpisodesBufferEntry:
    """Entry for episode buffer"""

    def __init__(self):
        # 存储每个时间步的视图（通常是环境的观测）
        self.views = []
        # 存储每个时间步的特征（可能是从视图中提取的特征）
        self.features = []
        # 存储每个时间步的动作。
        self.actions = []
        # 存储每个时间步的奖励
        self.rewards = []
        # 存储每个时间步的动作概率（可能用于策略梯度方法）
        self.probs = []
        # 一个布尔值，表示这个 episode 是否已经结束
        self.terminal = False

    # 添加数据
    def append(self, view, feature, action, reward, alive, probs=None):
        self.views.append(view.copy())
        self.features.append(feature.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        if probs is not None:
            self.probs.append(probs)
        if not alive:
            self.terminal = True


# 用于储存多个智能体的完整 episode 数据
class EpisodesBuffer(Buffer):
    """Replay buffer to store a whole episode for all agents
       one entry for one agent
    """

    def __init__(self, use_mean=False):
        super().__init__()

        # 一个字典，用于存储每个智能体的 episode 数据
        self.buffer = {}
        self.use_mean = use_mean

    def push(self, **kwargs):
        view, feature = kwargs['state']
        acts = kwargs['acts']
        rewards = kwargs['rewards']
        # 智能体存活状态
        alives = kwargs['alives']
        # 智能体id
        ids = kwargs['ids']

        if self.use_mean:
            probs = kwargs['prob']

        buffer = self.buffer
        # 返回一个长度等于len(view)，数值为0-len(view)的随机排序的数组，类似于[3, 1, 4, 0, 2]
        index = np.random.permutation(len(view))

        # 取出每一个智能体的缓存区EpisodesBufferEntry，添加状态
        for i in range(len(ids)):
            i = index[i]
            entry = buffer.get(ids[i])
            if entry is None:
                entry = EpisodesBufferEntry()
                buffer[ids[i]] = entry

            if self.use_mean:
                # EpisodesBufferEntry类的方法，针对这个智能体更新数据缓存
                entry.append(view[i],
                             feature[i],
                             acts[i],
                             rewards[i],
                             alives[i],
                             probs=probs[i])
            else:
                entry.append(view[i], feature[i], acts[i], rewards[i],
                             alives[i])

    def reset(self):
        """ clear replay buffer """
        self.buffer = {}

    # 返回所有存储的数据
    def episodes(self):
        """ get episodes """
        return self.buffer.values()


# 用于管理和存储单个智能体的交互数据
# 使用MetaBuffer环形缓存区来存储各类数据
class AgentMemory(object):

    def __init__(self, obs_shape, feat_shape, act_n, max_len, use_mean=False):
        self.obs0 = MetaBuffer(obs_shape, max_len)
        # 特征数据
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.use_mean = use_mean

        if self.use_mean:
            self.prob = MetaBuffer((act_n, ), max_len)

    # 添加各类数据
    def append(self, obs0, feat0, act, reward, alive, prob=None):
        self.obs0.append(np.array([obs0]))
        self.feat0.append(np.array([feat0]))
        self.actions.append(np.array([act], dtype=np.int32))
        self.rewards.append(np.array([reward]))
        self.terminals.append(np.array([not alive], dtype=np.bool))

        if self.use_mean:
            self.prob.append(np.array([prob]))

    # 取出该智能体的所有历史数据
    def pull(self):
        res = {
            'obs0': self.obs0.pull(),
            'feat0': self.feat0.pull(),
            'act': self.actions.pull(),
            'rewards': self.rewards.pull(),
            'terminals': self.terminals.pull(),
            'prob': None if not self.use_mean else self.prob.pull()
        }

        return res


# 用于管理多个智能体的交互数据。通过 AgentMemory 类来存储每个智能体的数据
class MemoryGroup(object):

    def __init__(self,
                 obs_shape,
                 feat_shape,
                 act_n,
                 max_len,
                 batch_size,
                 sub_len,
                 use_mean=False):
        # 存储所有智能体的id：AgentMemory对
        self.agent = dict()
        # 缓冲区最大长度
        self.max_len = max_len
        # batch批次大小
        self.batch_size = batch_size
        self.obs_shape = obs_shape
        self.feat_shape = feat_shape
        # 每个智能体的子缓冲区长度
        self.sub_len = sub_len
        self.use_mean = use_mean
        self.act_n = act_n

        # 初始化多个环形缓存区，一个缓存区存储一类数据
        self.obs0 = MetaBuffer(obs_shape, max_len)
        self.feat0 = MetaBuffer(feat_shape, max_len)
        self.actions = MetaBuffer((), max_len, dtype='int32')
        self.rewards = MetaBuffer((), max_len)
        # 存储终止状态数据，bool类型
        self.terminals = MetaBuffer((), max_len, dtype='bool')
        self.masks = MetaBuffer((), max_len, dtype='bool')
        if use_mean:
            self.prob = MetaBuffer((act_n, ), max_len)
        # 记录新添加的数据量
        self._new_add = 0

    # 传入数据（所有智能体的数据一齐存储）
    def _flush(self, **kwargs):
        # 将传入的数据添加到相应的 MetaBuffer缓存区中
        self.obs0.append(kwargs['obs0'])
        self.feat0.append(kwargs['feat0'])
        self.actions.append(kwargs['act'])
        self.rewards.append(kwargs['rewards'])
        self.terminals.append(kwargs['terminals'])

        if self.use_mean:
            self.prob.append(kwargs['prob'])

        # np.where()通过条件判断创建一个新数组：对于kwargs['terminals']中等于True的元素，新数组中对应的位置将被赋值为False，反之为True
        # 强化学习中，掩码通常用于标识有效的时间步，terminals为True表示终止状态，则说明这段数据是无效的，所以用mask表示遮蔽这段数据
        # 掩码的存在可以帮助在采样和训练过程中过滤掉无效的数据，确保算法只使用有效的时间步数据进行计算。
        mask = np.where(kwargs['terminals'] == True, False, True)
        # 为什么最后一个时间步的数据不被使用：强化学习中最后一个时间步的数据可能不完整或者不适用某些运算
        mask[-1] = False
        self.masks.append(mask)

    # 为每个智能体存储数据
    def push(self, **kwargs):
        for i, _id in enumerate(kwargs['ids']):
            if self.agent.get(_id) is None:
                self.agent[_id] = AgentMemory(self.obs_shape,
                                              self.feat_shape,
                                              self.act_n,
                                              self.sub_len,
                                              use_mean=self.use_mean)
            if self.use_mean:
                self.agent[_id].append(obs0=kwargs['state'][0][i],
                                       feat0=kwargs['state'][1][i],
                                       act=kwargs['acts'][i],
                                       reward=kwargs['rewards'][i],
                                       alive=kwargs['alives'][i],
                                       prob=kwargs['prob'][i])
            else:
                self.agent[_id].append(obs0=kwargs['state'][0][i],
                                       feat0=kwargs['state'][1][i],
                                       act=kwargs['acts'][i],
                                       reward=kwargs['rewards'][i],
                                       alive=kwargs['alives'][i])

    # 将各智能体的数据从各自的缓存区合并到主缓存区
    def tight(self):
        ids = list(self.agent.keys())
        # 对id序列随机打乱
        np.random.shuffle(ids)
        for ele in ids:
            tmp = self.agent[ele].pull()
            # _new_add表征新添加的数据量
            self._new_add += len(tmp['obs0'])
            self._flush(**tmp)
        # 清空智能体缓存区，这样做是为了在下一次数据收集时重新初始化每个智能体的缓冲区。
        self.agent = dict()  # clear

    # 随机选择batch个时间步，返回这些时间步的各项数据
    def sample(self):
        # 随机选择 batch_size 个索引，这些索引是针对self.obs0缓存区的
        idx = np.random.choice(self.nb_entries, size=self.batch_size)
        # 计算这些索引的下一个时间步索引
        next_idx = (idx + 1) % self.nb_entries

        # 将这些索引在多个类型的缓存区中对应的数据都取出来，obs0中存储所有智能体的每个时间步的观测数据
        obs = self.obs0.sample(idx)
        obs_next = self.obs0.sample(next_idx)
        feature = self.feat0.sample(idx)
        feature_next = self.feat0.sample(next_idx)
        actions = self.actions.sample(idx)
        rewards = self.rewards.sample(idx)
        dones = self.terminals.sample(idx)
        masks = self.masks.sample(idx)

        if self.use_mean:
            act_prob = self.prob.sample(idx)
            act_next_prob = self.prob.sample(next_idx)
            return obs, feature, actions, act_prob, obs_next, feature_next, act_next_prob, rewards, dones, masks
        else:
            return obs, feature, obs_next, feature_next, dones, rewards, actions, masks

    # 计算当前缓冲区中可以采样的有效批次数
    # （因为在强化学习中，尤其是使用经验回放（Experience Replay）技术时，数据是以批次（batch）的形式进行采样和训练的）
    def get_batch_num(self):
        # 打印缓冲区的长度和新添加的数据量
        print('\n[INFO] Length of buffer and new add:', len(self.obs0),
              self._new_add)
        # _new_add是最近一次数据合并后新增的数据量，计算这些新增数据可以采样的有效批次数
        # 乘以2是为了引入一定的冗余
        res = self._new_add * 2 // self.batch_size
        self._new_add = 0
        return res

    @property
    # 返回缓存区的长度
    def nb_entries(self):
        return len(self.obs0)


# 用于管理 TensorFlow 的日志记录和摘要操作
# 在训练过程中记录各种指标
class SummaryObj:
    """
    Define a summary holder
    """

    def __init__(self, log_dir, log_name, n_group=1):
        self.name_set = set()

        # 初始化一个计算图graph
        # 创建一个独立的计算环境，在这个环境中可以定义各种计算操作、变量和常量等。这样可以更好地组织和管理复杂的计算任务，尤其是在处理多个不同的模型或者复杂的计算流程时。
        # graph会作为会话的输入创建会话 sess = tf.Session(graph=graph)
        self.gra = tf.Graph()
        self.n_group = n_group

        # 创建日志文件夹
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # session参数设置：
        """
        allow_soft_placement=True：允许 TensorFlow 自动选择设备。
        log_device_placement=False：不记录设备分配日志。
        gpu_options.allow_growth=True：动态分配 GPU 内存。
        """
        sess_config = tf.ConfigProto(allow_soft_placement=True,
                                     log_device_placement=False)
        sess_config.gpu_options.allow_growth = True

        # with self.gra.as_default():是在计算图gra中定义一些操作
        with self.gra.as_default():
            self.sess = tf.Session(graph=self.gra, config=sess_config)
            # 创建一个 FileWriter 对象 self.train_writer，用于将摘要信息写入指定的日志目录
            self.train_writer = tf.summary.FileWriter(
                log_dir + "/" + log_name, graph=tf.get_default_graph())
            #  初始化 self.gra 计算图中的所有全局变量
            self.sess.run(tf.global_variables_initializer())

    # 用于注册一组摘要操作
    # 摘要操作通常用于在训练过程中记录和监控各种指标（如损失、准确率等）
    def register(self, name_list):
        """Register summary operations with a list contains names for these operations

        Parameters
        ----------
        name_list: list, contains name whose type is str
        """

        # 确保在当前的计算图self.gra中定义操作
        with self.gra.as_default():
            for name in name_list:
                if name in self.name_set:
                    raise Exception(
                        "You cannot define different operations with same name: `{}`"
                        .format(name))
                self.name_set.add(name)

                # 使用setattr动态地为对象添加属性
                """
                setattr(object, name, value)
                object：要设置属性的对象。
                name：属性名，可以是字符串形式。
                value：要设置的属性值。
                """
                # 创建摘要属性
                setattr(self, name, [
                    tf.placeholder(tf.float32,
                                   shape=None,
                                   name='Agent_{}_{}'.format(i, name))
                    for i in range(self.n_group)
                ])
                # 创建监控属性：监控各种指标（如损失、准确率等），监控从摘要属性中读取
                setattr(self, name + "_op", [
                    tf.summary.scalar('Agent_{}_{}_op'.format(i, name),
                                      getattr(self, name)[i])
                    for i in range(self.n_group)
                ])

    # 训练过程中写入摘要数据
    def write(self, summary_dict, step):
        """Write summary related to a certain step

        Parameters
        ----------
        summary_dict: dict, summary value dict
        step: int, global step
        """

        assert isinstance(summary_dict, dict)

        # summary_dict: 一个字典，键是摘要操作的名称，值是对应的值（列表）
        for key, value in summary_dict.items():
            #  检查键是否在 self.name_set 中，如果不在则抛出异常
            if key not in self.name_set:
                raise Exception("Undefined operation: `{}`".format(key))
            if isinstance(value, list):

                # 遍历每个组（n_group），并使用 self.sess.run 执行相应的摘要操作，将结果写入 self.train_writer 中
                # add_summary 方法将 session.run 执行摘要操作的结果写入到日志文件中
                for i in range(self.n_group):
                    self.train_writer.add_summary(
                        self.sess.run(
                            # 获取摘要操作
                            getattr(self, key + "_op")[i],
                            # feed_dict={...} 是一个字典，用于将实际的值传递给 placeholder
                            feed_dict={getattr(self, key)[i]: value[i]}),
                        global_step=step)
            # 如果值不是列表，则只处理第一个组group的摘要操作
            else:
                self.train_writer.add_summary(self.sess.run(
                    getattr(self, key + "_op")[0],
                    feed_dict={getattr(self, key)[0]: value}),
                                              global_step=step)


# Runner类：用于在一个网格世界环境中运行和训练强化学习模型
class Runner(object):

    def __init__(self,
                 sess,
                 env,
                 handles,
                 map_size,
                 max_steps,
                 models,
                 play_handle,
                 render_every=None,
                 save_every=None,
                 tau=None,
                 log_name=None,
                 log_dir=None,
                 model_dir=None,
                 train=False):
        """Initialize runner

        Parameters
        ----------
        sess: tf.Session
            session
        env: magent.GridWorld
            environment handle
        handles: list
            group handles
        map_size: int
            map size of grid world
        max_steps: int
            the maximum of stages in a episode
        render_every: int 渲染环境的间隔
            render environment interval
        save_every: int 保存模型的间隔
            states the interval of evaluation for self-play update
        models: list 包含模型的列表
            contains models
        play_handle: method like 运行游戏的方法的handle
            run game
        tau: float 自我对弈更新的指数
            tau index for self-play update
        log_name: str 日志目录的名称
            define the name of log dir
        log_dir: str 日志目录
            donates the directory of logs
        model_dir: str 模型目录
            donates the dircetory of models
        """
        self.env = env
        self.models = models
        self.max_steps = max_steps
        self.handles = handles
        self.map_size = map_size
        self.render_every = render_every
        self.save_every = save_every
        self.play = play_handle
        self.model_dir = model_dir
        self.train = train

        if self.train:
            # 创建 SummaryObj 对象，用于记录训练过程中的摘要数据
            self.summary = SummaryObj(log_name=log_name, log_dir=log_dir)

            # 注册一些摘要项目，如平均奖励、总奖励、击杀数、奖励和、击杀总数
            summary_items = [
                'ave_agent_reward', 'total_reward', 'kill', "Sum_Reward",
                "Kill_Sum"
            ]
            self.summary.register(summary_items)  # summary register
            self.summary_items = summary_items

            assert isinstance(sess, tf.Session)
            # name_scope为该model定义的变量作用域，这里确保两个模型的变量作用域不同
            assert self.models[0].name_scope != self.models[1].name_scope
            self.sess = sess

            # 获取两个模型的各自变量域中的所有变量
            l_vars, r_vars = self.models[0].vars, self.models[1].vars
            # 确保变量长度相等
            assert len(l_vars) == len(r_vars)

            # self.sp_op存储了一系列软更新操作，通过调用self.sess.run(self.sp_op)就可以实现r_vars的更新
            # 这里l_vars定义为主模型（主网络），通常频繁更新，这是主要用于学习和更新策略的模型。它根据当前的经验（如状态、动作、奖励）进行频繁的参数更新，以优化策略或价值函数
            # r_vars定义为目标网络，通常缓慢更新，r_vars的参数使用下面的式子加权更新
            self.sp_op = [
                tf.assign(r_vars[i], (1. - tau) * l_vars[i] + tau * r_vars[i])
                for i in range(len(l_vars))
            ]

            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)

    # 用于执行强化学习的训练或评估的一个回合
    # variant_eps：探索率（epsilon），用于控制在当前训练回合中的探索和利用平衡
    # iteration：当前训练的迭代次数
    def run(self, variant_eps, iteration, win_cnt=None):
        info = {'main': None, 'opponent': None}

        # pass
        # 初始化字典信息
        # info：一个字典，包含主模型和对手模型的统计信息，包括平均奖励、总奖励和击杀数
        # "mian"为我方，"opponent"为敌方
        info['main'] = {'ave_agent_reward': 0., 'total_reward': 0., 'kill': 0.}
        info['opponent'] = {
            'ave_agent_reward': 0.,
            'total_reward': 0.,
            'kill': 0.
        }

        # self.play是在创建Runner对象时输入的play_handle（一个函数句柄）
        # 进行一个回合的仿真和模型的训练
        """
        max_nums：所有group的最大单位数
        nums：所有group在回合结束时的单位总数
        agent_r_records：所有group在本回合内的平均reward
        total_rewards：所有单位在本回合内的总reward
        """
        max_nums, nums, agent_r_records, total_rewards = self.play(
            env=self.env,
            n_round=iteration,
            map_size=self.map_size,
            max_steps=self.max_steps,
            handles=self.handles,
            models=self.models,
            print_every=50,
            eps=variant_eps,
            render=(iteration + 1) %
            self.render_every if self.render_every > 0 else False,
            train=self.train)

        # 更新info表中的总奖励、杀敌数、平均奖励
        for i, tag in enumerate(['main', 'opponent']):
            info[tag]['total_reward'] = total_rewards[i]
            info[tag]['kill'] = max_nums[i] - nums[1 - i]
            info[tag]['ave_agent_reward'] = agent_r_records[i]

        # 如果在训练模式下，进行模型参数的更新
        if self.train:
            # 打印友方智能体的info
            print('\n[INFO] {}'.format(info['main']))

            # if self.save_every and (iteration + 1) % self.save_every == 0:
            # 如果友方智能体的总奖励大于敌方智能体，进行自对弈更新
            # 注：在self.play中已经更新了model[0]的参数，这里用自对弈方式更新model[1]的参数，一般来说model[1]可能是旧版本或已经可用的模型
            # 这样可以让model[1]学习model[0]的策略，增大模型1的战斗力，反过来增大model[0]受挑战的强度
            if info['main']['total_reward'] > info['opponent']['total_reward']:
                print(Color.INFO.format('\n[INFO] Begin self-play Update ...'))

                # 运行 TensorFlow 会话中的自对弈更新操作
                # tf.assign(r_vars[i], (1. - tau) * l_vars[i] + tau * r_vars[i])
                self.sess.run(self.sp_op)
                print(Color.INFO.format('[INFO] Self-play Updated!\n'))

                print(Color.INFO.format('[INFO] Saving model ...'))
                # 调用save函数保存模型权重
                self.models[0].save(self.model_dir + '-0', iteration)
                self.models[1].save(self.model_dir + '-1', iteration)

                # 将该次仿真的结果写入摘要
                self.summary.write(info['main'], iteration)
        else:
            # 如果不在训练模式下，记录该次对抗的结果
            print('\n[INFO] {0} \n {1}'.format(info['main'], info['opponent']))
            if info['main']['kill'] > info['opponent']['kill']:
                win_cnt['main'] += 1
            elif info['main']['kill'] < info['opponent']['kill']:
                win_cnt['opponent'] += 1
            else:
                win_cnt['main'] += 1
                win_cnt['opponent'] += 1
