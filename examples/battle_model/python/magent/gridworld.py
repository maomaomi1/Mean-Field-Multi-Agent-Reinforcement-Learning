"""gridworld interface"""
# 这将进行绝对导入
from __future__ import absolute_import

import ctypes
import os
import importlib

import numpy as np

from .c_lib import _LIB, as_float_c_array, as_int32_c_array
from .environment import Environment


class GridWorld(Environment):
    # constant
    # view缓存区索引
    OBS_INDEX_VIEW = 0
    # feature缓存区索引
    OBS_INDEX_HP = 1

    def __init__(self, config, **kwargs):
        """
        Parameters
        ----------
        config: str or Config Object
            if config is a string, then it is a name of builtin config,
                builtin config are stored in python/magent/builtin/config
                kwargs are the arguments to the config
            if config is a Config Object, then parameters are stored in that object
        """
        Environment.__init__(self)

        # if is str, load built in configuration
        # 如果config是str，导入内置config
        if isinstance(config, str):
            # built-in config are stored in python/magent/builtin/config
            try:
                # config是str，对战中导入的库为python/magent/builtin/config/battle.py
                demo_game = importlib.import_module('magent.builtin.config.' +
                                                    config)
                # getattr用于获取一个对象的属性，这里是一个'get_config'函数，传入(**kwargs)参数
                config = getattr(demo_game, 'get_config')(**kwargs)
            except AttributeError:
                raise BaseException('unknown built-in game "' + config + '"')

        # create new game
        # 使用 C 接口 _LIB.env_new_game 创建一个新的游戏实例，并将其存储在 self.game 中
        # ctypes 是 Python 的一个外部函数库，允许调用 C 函数和使用 C 数据类型

        # 创建了一个 ctypes.c_void_p 类型的变量 game，它是一个指向任意类型的 C 指针
        game = ctypes.c_void_p()
        # game是一个指向环境句柄的指针，用于存储创建的游戏实例
        _LIB.env_new_game(ctypes.byref(game), b"GridWorld")
        self.game = game

        # set global configuration
        config_value_type = {
            'map_width': int,
            'map_height': int,
            'food_mode': bool,
            'turn_mode': bool,
            'minimap_mode': bool,
            'revive_mode': bool,
            'goal_mode': bool,
            'embedding_size': int,
            'render_dir': str,
        }

        # 将全局配置转换成C++的数据类型，并导入game的设置中
        # 全局配置只有"map_width"、"map_height"、"minimap_mode"、"embedding_size"
        for key in config.config_dict:
            value_type = config_value_type[key]
            if value_type is int:
                _LIB.env_config_game(
                    self.game, key.encode("ascii"),
                    ctypes.byref(ctypes.c_int(config.config_dict[key])))
            elif value_type is bool:
                _LIB.env_config_game(
                    self.game, key.encode("ascii"),
                    ctypes.byref(ctypes.c_bool(config.config_dict[key])))
            elif value_type is float:
                _LIB.env_config_game(
                    self.game, key.encode("ascii"),
                    ctypes.byref(ctypes.c_float(config.config_dict[key])))
            elif value_type is str:
                _LIB.env_config_game(self.game, key.encode("ascii"),
                                     ctypes.c_char_p(config.config_dict[key]))

        # 注册代理类型
        for name in config.agent_type_dict:
            # 取出所有作战类型
            """
            type_args包括：
            'width': 1,
            'length': 1,
            'hp': 10,
            'speed': 2,
            'view_range': gw.CircleRange(6),
            'attack_range': gw.CircleRange(1.5),
            'damage': 2,
            'step_recover': 0.1,
            'step_reward': -0.005,
            'kill_reward': 5,
            'dead_penalty': -0.1,
            'attack_penalty': -0.1,
            """
            type_args = config.agent_type_dict[name]

            # special pre-process for view range and attack range
            # 预处理视野范围和攻击范围（不用看）
            for key in [x for x in type_args.keys()]:
                if key == "view_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["view_radius"] = val.radius
                    type_args["view_angle"] = val.angle
                elif key == "attack_range":
                    val = type_args[key]
                    del type_args[key]
                    type_args["attack_radius"] = val.radius
                    type_args["attack_angle"] = val.angle

            length = len(type_args)
            keys = (ctypes.c_char_p *
                    length)(*[key.encode("ascii") for key in type_args.keys()])
            values = (ctypes.c_float * length)(*type_args.values())

            _LIB.gridworld_register_agent_type(self.game, name.encode("ascii"),
                                               length, keys, values)

        # serialize event expression, send to C++ engine
        # 序列化事件表达式，并发送到 C++ 引擎
        self._serialize_event_exp(config)

        # init group handles
        # 初始化群组句柄，self.group_handles中的每个元素指向config.groups中的每个元素（单位类型）
        self.group_handles = []
        for item in config.groups:
            handle = ctypes.c_int32()
            _LIB.gridworld_new_group(self.game, item.encode("ascii"),
                                     ctypes.byref(handle))
            self.group_handles.append(handle)

        # init observation buffer (for acceleration)
        # 初始化观察缓冲区
        self._init_obs_buf()

        # init view space, feature space, action space
        # 初始化视野空间、特征空间和动作空间
        
        ### self.view_space = （height,width,n_channel),height=width表示这一类agenttype的探测圆形的直径，n_channel表示通道数wall + additional + (has, hp) + (has, hp)即各种类型指标，比如只观测数量、生命值、wall，那么n_channel=3
        ### self.feature_space = （features)，特征空间只有一维，features = embedding_size+AgentType.action_space.size() + 1
        ### self.action_space = (AgentType.action_space)，特征空间只有一维，包括可以移动的点和可以攻击的点
        self.view_space = {}
        self.feature_space = {}
        self.action_space = {}
        # 创建一个包含3个元素的一维数组，但是未初始化，需要后续填充
        buf = np.empty((3, ), dtype=np.int32)
        for handle in self.group_handles:

            # 传递游戏实例 self.game、群组句柄 handle，请求的信息为视野空间，存储到缓存区buf中，因此buf为视野空间信息
            _LIB.env_get_info(
                self.game, handle, b"view_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            # 这里的handle.value是一个整数值，可以认为是单位类型的id
            self.view_space[handle.value] = (buf[0], buf[1], buf[2])
            _LIB.env_get_info(
                self.game, handle, b"feature_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            # 为什么要加逗号，是因为要创建元组，(buf[0], )是一个包含单个元素的元组。这个逗号是必需的
            # 例如，(1)在 Python 中被视为整数1，而(1,)才是一个包含整数1的元组
            self.feature_space[handle.value] = (buf[0], )
            _LIB.env_get_info(
                self.game, handle, b"action_space",
                buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
            self.action_space[handle.value] = (buf[0], )

    # 重置环境
    def reset(self):
        """reset environment"""
        _LIB.env_reset(self.game)

    # 为场景中添加墙壁：障碍物
    def add_walls(self, method, **kwargs):
        """add wall to environment

        Parameters
        ----------
        method: str
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_walls(method="random", n=1000)

        # add 3 walls to (1,2), (4,5) and (9, 8) in map
        >>> env.add_walls(method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        # handle = -1 for walls
        kwargs["dir"] = 0
        # agents类型为-1（第一个输入）代表墙壁
        self.add_agents(-1, method, **kwargs)

    # ====== AGENT ======
    # 在环境中注册一个新的群组group，返回新群组的句柄
    def new_group(self, name):
        """register a new group into environment"""
        handle = ctypes.c_int32()
        _LIB.gridworld_new_group(self.game,
                                 ctypes.c_char_p(name.encode("ascii")),
                                 ctypes.byref(handle))
        return handle

    # 往handle代表的group中按指定方法添加智能体
    # 如果handle为-1,代表添加墙体，其他则是具体group的handle
    def add_agents(self, handle, method, **kwargs):
        """add agents to environment

        Parameters
        ----------
        handle: group handle
        method: str
        有多种添加智能体的方法
            can be 'random' or 'custom'
            if method is 'random', then kwargs["n"] is a int
            if method is 'custom', then kwargs["pos"] is a list of coordination

        Examples
        --------
        # add 1000 walls randomly
        >>> env.add_agents(handle, method="random", n=1000)

        # add 3 agents to (1,2), (4,5) and (9, 8) in map
        >>> env.add_agents(handle, method="custom", pos=[(1,2), (4,5), (9,8)])
        """
        if method == "random":
            _LIB.gridworld_add_agents(self.game, handle, int(kwargs["n"]),
                                      b"random", 0, 0, 0)
        elif method == "custom":
            n = len(kwargs["pos"])
            pos = np.array(kwargs["pos"], dtype=np.int32)
            if len(pos) <= 0:
                return
            if pos.shape[1] == 3:  # if has dir
                xs, ys, dirs = pos[:, 0], pos[:, 1], pos[:, 2]
            else:  # if do not has dir, use zero padding
                xs, ys, dirs = pos[:, 0], pos[:, 1], np.zeros((n, ),
                                                              dtype=np.int32)
            # copy again, to make these arrays continuous in memory
            xs, ys, dirs = np.array(xs), np.array(ys), np.array(dirs)
            _LIB.gridworld_add_agents(self.game, handle, n, b"custom",
                                      as_int32_c_array(xs),
                                      as_int32_c_array(ys),
                                      as_int32_c_array(dirs))
        elif method == "fill":
            x, y = kwargs["pos"][0], kwargs["pos"][1]
            width, height = kwargs["size"][0], kwargs["size"][1]
            # 获取单位的方向，没有则用0填充
            dir = kwargs.get("dir", np.zeros_like(x))
            bind = np.array([x, y, width, height, dir], dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0, b"fill",
                                      as_int32_c_array(bind), 0, 0, 0)
        elif method == "maze":
            # TODO: implement maze add
            x_start, y_start, x_end, y_end = kwargs["pos"][0], kwargs["pos"][
                1], kwargs["pos"][2], kwargs["pos"][3]
            thick = kwargs["pos"][4]
            bind = np.array([x_start, y_start, x_end, y_end, thick],
                            dtype=np.int32)
            _LIB.gridworld_add_agents(self.game, handle, 0, b"maze",
                                      as_int32_c_array(bind), 0, 0, 0)
        else:
            print("Unknown type of position")
            exit(-1)

    # ====== RUN ======
    # 从环境中获取group的观测缓存区
    def _get_obs_buf(self, group, key, shape, dtype):
        """get buffer to receive observation from c++ engine"""
        obs_buf = self.obs_bufs[key]
        if group in obs_buf:
            ret = obs_buf[group]
            # shape：缓存区的形状
            # 检查缓冲区形状是否匹配，若不匹配则调整大小
            if shape != ret.shape:
                ret.resize(shape, refcheck=False)
        else:
            # 若不存在group的缓存区，添加一个空的缓存区
            ret = obs_buf[group] = np.empty(shape=shape, dtype=dtype)

        return ret

    def _init_obs_buf(self):
        """init observation buffer"""
        self.obs_bufs = []
        self.obs_bufs.append({})
        self.obs_bufs.append({})

    # 获取一个group的观测数据，包括视图和特征，handle为group的句柄，包括group中所有智能体的observation
    # ！！！！！！这个返回的缓存区中存储的是当前步的信息，因为每次调用函数会初始化缓存区，缓存区不包含历史信息
    def get_observation(self, handle):
        """ get observation of a whole group

        Parameters
        ----------
        handle : group handle

        Returns
        -------
        obs : tuple (views, features)
        n代表group中的智能体个数
            views is a numpy array, whose shape is n * view_width * view_height * n_channel（图的长、宽和通道数）
            features is a numpy array, whose shape is n * feature_size
            for agent i, (views[i], features[i]) is its observation at this step
        """
        # view_space并不是view数据，而是view的维度，即view_width * view_height * n_channel，feature_space同理
        view_space = self.view_space[handle.value]
        feature_space = self.feature_space[handle.value]
        no = handle.value

        # group中的智能体个数
        n = self.get_num(handle)
        # 获取缓存区，其实是创建空的数据表，大小为(n, ) + view_space，n为智能体个数
        view_buf = self._get_obs_buf(no, self.OBS_INDEX_VIEW,
                                     (n, ) + view_space, np.float32)
        feature_buf = self._get_obs_buf(no, self.OBS_INDEX_HP,
                                        (n, ) + feature_space, np.float32)

        # 创建指向缓冲区的指针数组
        bufs = (ctypes.POINTER(ctypes.c_float) * 2)()
        bufs[0] = as_float_c_array(view_buf)
        bufs[1] = as_float_c_array(feature_buf)

        # 调用 C 库函数获取观察数据
        # 观察到的数据被存储在缓存区bufs中，也就是更新了view_buf, feature_buf
        _LIB.env_get_observation(self.game, handle, bufs)

        return view_buf, feature_buf

    # 为所有单位设置动作，actions为数组，包含每个agent的动作
    def set_action(self, handle, actions):
        """ set actions for whole group

        Parameters
        ----------
        handle: group handle
        actions: numpy array
            the dtype of actions must be int32
        """
        # actions必须是整型多维数组
        assert isinstance(actions, np.ndarray)
        assert actions.dtype == np.int32
        _LIB.env_set_action(
            self.game, handle,
            actions.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))

    # 为每个单位选择action后，仿真推进一步
    # 返回值：游戏是否已经结束
    def step(self):
        """simulation one step after set actions

        Returns
        -------
        done: bool
            whether the game is done
        """
        done = ctypes.c_int32()
        # 看gridWorld.cc中的step函数的具体实现
        _LIB.env_step(self.game, ctypes.byref(done))
        return bool(done)

    # 获取该group中每个单位的reward，输出为n个元素的元组
    def get_reward(self, handle):
        """ get reward for a whole group

        Returns
        -------
        rewards: numpy array (float32)
            reward for all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n, ), dtype=np.float32)
        _LIB.env_get_reward(self.game, handle,
                            buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    # 清除死亡单位
    def clear_dead(self):
        """ clear dead agents in the engine
        must be called after step()
        """
        _LIB.gridworld_clear_dead(self.game)

    # ====== INFO ======
    def get_handles(self):
        """ get all group handles in the environment """
        return self.group_handles

    # 获取group中的智能体个数
    def get_num(self, handle):
        """ get the number of agents in a group"""
        num = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b'num', ctypes.byref(num))
        return num.value

    # 获取动作空间，返回的不是信息而是维度
    def get_action_space(self, handle):
        """get action space

        Returns
        -------
        action_space : tuple
        """
        return self.action_space[handle.value]

    # 获取视图空间，返回的不是信息而是维度
    def get_view_space(self, handle):
        """get view space

        Returns
        -------
        view_space : tuple
        """
        return self.view_space[handle.value]

    # 获取特征空间，返回的不是信息而是维度
    def get_feature_space(self, handle):
        """ get feature space

        Returns
        -------
        feature_space : tuple
        """
        return self.feature_space[handle.value]

    # 获取指定handle组中所有单位的id，返回填充好的NumPy数组 buf
    def get_agent_id(self, handle):
        """ get agent id

        Returns
        -------
        ids : numpy array (int32)
            id of all the agents in the group
        """
        n = self.get_num(handle)
        buf = np.empty((n, ), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"id",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    # 返回指定group中所有单位的存活状态，返回存活状态的NumPy数组 buf（元素bool类型）
    def get_alive(self, handle):
        """ get alive status of agents in a group

        Returns
        -------
        alives: numpy array (bool)
            whether the agents are alive
        """
        n = self.get_num(handle)
        buf = np.empty((n, ), dtype=np.bool)
        _LIB.env_get_info(self.game, handle, b"alive",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)))
        return buf

    # 返回组中所有单位的位置，返回buf结构(n,2)因为只考虑xy轴
    def get_pos(self, handle):
        """ get position of agents in a group

        Returns
        -------
        pos: numpy array (int)
            the shape of pos is (n, 2)
        """
        n = self.get_num(handle)
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, handle, b"pos",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    # 已经被弃用的方法，用于获取均值信息
    def get_mean_info(self, handle):
        """ deprecated """
        buf = np.empty(2 + self.action_space[handle.value][0],
                       dtype=np.float32)
        _LIB.env_get_info(self.game, handle, b"mean_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    # 获取与视野范围相同大小的矩阵，指示可攻击的点
    def get_view2attack(self, handle):
        """ get a matrix with the same size of view_range,
            if element >= 0, then it means it is a attackable point, and the corresponding
                                    action number is the value of that element
        Returns
        -------
        attack_back: int
        buf: numpy array
            map attack action into view
        """
        """
        获取一个与视野范围相同大小的矩阵。如果矩阵中的元素大于或等于0，则表示该点是可攻击的点，元素的值对应动作的编号。
        """
        # 获取视野范围大小（长宽）
        size = self.get_view_space(handle)[0:2]
        # 创建缓冲区
        buf = np.empty(size, dtype=np.int32)
        # 创建攻击基数变量
        attack_base = ctypes.c_int32()
        _LIB.env_get_info(self.game, handle, b"view2attack",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        _LIB.env_get_info(self.game, handle, b"attack_base",
                          ctypes.byref(attack_base))

        # 返回值：1、attack_base.value：攻击基数，是一个从游戏环境中获取的整数值，可能用于计算实际攻击力、作为动作编号的偏移量或标识特定攻击类型。
        #         2、buf：获取与视野范围相同大小的矩阵，指示可攻击的点
        return attack_base.value, buf

    # 将全局地图压缩为指定大小的缩略图the shape (n_group + 1, height, width)
    # 意思是每个group会对应一个(height, width)
    # 以(height, width)矩阵的形式来表示地图信息，每个元素代表什么？
    def get_global_minimap(self, height, width):
        """ compress global map into a minimap of given size
        Parameters
        ----------
        height: int
            the height of minimap
        width:  int
            the width of minimap

        Returns
        -------
        minimap : numpy array
            the shape (n_group + 1, height, width)
        """
        buf = np.empty((height, width, len(self.group_handles)),
                       dtype=np.float32)
        buf[0, 0, 0] = height
        buf[0, 0, 1] = width
        _LIB.env_get_info(self.game, -1, b"global_minimap",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return buf

    # 设置游戏引擎的随机种子
    def set_seed(self, seed):
        """ set random seed of the engine"""
        _LIB.env_config_game(self.game, b"seed",
                             ctypes.byref(ctypes.c_int(seed)))

    # ====== RENDER ======
    # 设置保存渲染文件的目录
    def set_render_dir(self, name):
        """ set directory to save render file"""
        if not os.path.exists(name):
            os.mkdir(name)
        _LIB.env_config_game(self.game, b"render_dir", name.encode("ascii"))

    # 渲染一个仿真步step
    def render(self):
        """ render a step """
        _LIB.env_render(self.game)

    # 获取组（所有groups）的相关信息（私有方法，用于交互式应用）
    # 每个group对应5个元素，最终返回(n,5)，暂未知5个元素分别代表什么
    def _get_groups_info(self):
        """ private method, for interactive application"""
        n = len(self.group_handles)

        # 每个group对应5个元素，最终返回(n,5)，暂未知5个元素分别代表什么
        buf = np.empty((n, 5), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"groups_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        return buf

    # 获取墙壁的相关信息
    def _get_walls_info(self):
        """ private method, for interactive application"""
        n = 100 * 100
        buf = np.empty((n, 2), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"walls_info",
                          buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        # 前n项代表墙壁，返回所有的墙壁的信息
        n = buf[0, 0]  # the first line is the number of walls
        return buf[1:1 + n]

    # 获取渲染窗口内的代理和攻击事件信息（私有方法，用于交互式应用）
    # 输出1：所有group中所有智能体的信息（id,位置，状态等）
    # 输出2：所有攻击时间events
    def _get_render_info(self, x_range, y_range):
        """ private method, for interactive application"""
        n = 0
        # 获取所有group的智能体总和n
        for handle in self.group_handles:
            n += self.get_num(handle)

        # 创建缓冲区
        buf = np.empty((n + 1, 4), dtype=np.int32)
        # 设置缓冲区的第一行，用于指定渲染窗口的范围
        buf[0] = x_range[0], y_range[0], x_range[1], y_range[1]
        _LIB.env_get_info(self.game, -1, b"render_window_info",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))

        # the first line is for the number of agents in the window range
        ###### 解析缓冲区数据 #######

        # 第一行 buf[0] 包含窗口范围内的代理数量 agent_ct 和攻击事件数量 attack_event_ct
        info_line = buf[0]
        agent_ct, attack_event_ct = info_line[0], info_line[1]
        # 取出info部分
        buf = buf[1:1 + info_line[0]]

        agent_info = {}
        for item in buf:
            # item的结构应该是[agent_id,info1,info2,info3]
            # 转成agent_name：info的字典形式
            agent_info[item[0]] = [item[1], item[2], item[3]]

        # 攻击事件info为攻击事件数量（attack_event_ct，3）维度的结构
        buf = np.empty((attack_event_ct, 3), dtype=np.int32)
        _LIB.env_get_info(self.game, -1, b"attack_event",
                          buf.ctypes.data_as(ctypes.POINTER((ctypes.c_int32))))
        attack_event = buf

        return agent_info, attack_event

    # 删除游戏实例以释放资源
    def __del__(self):
        _LIB.env_delete_game(self.game)

    # ====== SPECIAL RULE ======
    # 设置目标（已弃用的方法）
    def set_goal(self, handle, method, *args, **kwargs):
        """ deprecated """
        if method == "random":
            _LIB.gridworld_set_goal(self.game, handle, b"random", 0, 0)
        else:
            raise NotImplementedError

    # ====== PRIVATE ======
    # 将事件表达式序列化并发送到游戏引擎
    # 也就是将发生的事件送入游戏引擎的中间操作，不用看
    def _serialize_event_exp(self, config):
        """serialize event expression and sent them to game engine"""
        game = self.game

        # collect agent symbol
        symbol2int = {}
        config.symbol_ct = 0

        def collect_agent_symbol(node, config):
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_agent_symbol(item, config)
                elif isinstance(item, AgentSymbol):
                    if item not in symbol2int:
                        symbol2int[item] = config.symbol_ct
                        config.symbol_ct += 1

        for rule in config.reward_rules:
            on = rule[0]
            receiver = rule[1]
            for symbol in receiver:
                if symbol not in symbol2int:
                    symbol2int[symbol] = config.symbol_ct
                    config.symbol_ct += 1
            collect_agent_symbol(on, config)

        # collect event node
        event2int = {}
        config.node_ct = 0

        def collect_event_node(node, config):
            if node not in event2int:
                event2int[node] = config.node_ct
                config.node_ct += 1
            for item in node.inputs:
                if isinstance(item, EventNode):
                    collect_event_node(item, config)

        for rule in config.reward_rules:
            collect_event_node(rule[0], config)

        # send to C++ engine
        for sym in symbol2int:
            no = symbol2int[sym]
            _LIB.gridworld_define_agent_symbol(game, no, sym.group, sym.index)

        for event in event2int:
            no = event2int[event]
            inputs = np.zeros_like(event.inputs, dtype=np.int32)
            for i, item in enumerate(event.inputs):
                if isinstance(item, EventNode):
                    inputs[i] = event2int[item]
                elif isinstance(item, AgentSymbol):
                    inputs[i] = symbol2int[item]
                else:
                    inputs[i] = item
            n_inputs = len(inputs)
            _LIB.gridworld_define_event_node(game, no, event.op,
                                             as_int32_c_array(inputs),
                                             n_inputs)

        for rule in config.reward_rules:
            # rule = [on, receiver, value, terminal]
            on = event2int[rule[0]]

            receiver = np.zeros_like(rule[1], dtype=np.int32)
            for i, item in enumerate(rule[1]):
                receiver[i] = symbol2int[item]
            if len(rule[2]) == 1 and rule[2][0] == 'auto':
                value = np.zeros(receiver, dtype=np.float32)
            else:
                value = np.array(rule[2], dtype=np.float32)
            n_receiver = len(receiver)
            _LIB.gridworld_add_reward_rule(game, on,
                                           as_int32_c_array(receiver),
                                           as_float_c_array(value), n_receiver,
                                           rule[3])


'''
the following classes are for reward description
'''


# 表示事件表达式的抽象语法树 (AST) 节点
class EventNode:
    """an AST node of the event expression"""
    # 定义不同的操作符常量
    OP_AND = 0
    OP_OR = 1
    OP_NOT = 2

    # 定义不同的事件event类型
    OP_KILL = 3
    OP_AT = 4
    OP_IN = 5
    OP_COLLIDE = 6
    OP_ATTACK = 7
    OP_DIE = 8
    OP_IN_A_LINE = 9
    OP_ALIGN = 10

    # can extend more operation below

    def __init__(self):
        # for non-leaf node
        # 操作符，默认为None
        self.op = None
        # for leaf node
        # 谓词（主体的动作）
        self.predicate = None

        # 输入列表，用于存储子节点或参数
        self.inputs = []

    # 重写了该对象的调用方法
    # 比如有一个实例Event，Event(*args)就等同于Event.__call__(*args)
    def __call__(self, subject, predicate, *args):
        # subject：事件的主体

        # 首先创建一个新的 EventNode 对象，并将传入的谓词赋值给 node.predicate
        node = EventNode()
        node.predicate = predicate
        if predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        elif predicate == 'at':
            node.op = EventNode.OP_AT
            # coor为坐标
            coor = args[0]
            node.inputs = [subject, coor[0], coor[1]]
        elif predicate == 'in':
            node.op = EventNode.OP_IN
            coor = args[0]
            x1, y1 = min(coor[0][0], coor[1][0]), min(coor[0][1], coor[1][1])
            x2, y2 = max(coor[0][0], coor[1][0]), max(coor[0][1], coor[1][1])
            node.inputs = [subject, x1, y1, x2, y2]
        elif predicate == 'attack':
            node.op = EventNode.OP_ATTACK
            node.inputs = [subject, args[0]]
        elif predicate == 'kill':
            node.op = EventNode.OP_KILL
            node.inputs = [subject, args[0]]
        # collide冲突碰撞
        elif predicate == 'collide':
            node.op = EventNode.OP_COLLIDE
            node.inputs = [subject, args[0]]
        elif predicate == 'die':
            node.op = EventNode.OP_DIE
            node.inputs = [subject]
        elif predicate == 'in_a_line':
            node.op = EventNode.OP_IN_A_LINE
            node.inputs = [subject]
        # align调整、使一致
        elif predicate == 'align':
            node.op = EventNode.OP_ALIGN
            node.inputs = [subject]
        else:
            raise Exception("invalid predicate of event " + predicate)
        return node

    # 重载了逻辑操作符 &、| 和 ~，分别对应 AND、OR 和 NOT 操作
    """     
    obj1 = EventNode()
    obj2 = EventNode()
    result = obj1 & obj2
    """

    def __and__(self, other):
        node = EventNode()
        node.op = EventNode.OP_AND
        node.inputs = [self, other]
        return node

    def __or__(self, other):
        node = EventNode()
        node.op = EventNode.OP_OR
        node.inputs = [self, other]
        return node

    def __invert__(self):
        node = EventNode()
        node.op = EventNode.OP_NOT
        node.inputs = [self]
        return node


Event = EventNode()


# AgentSymbol 类表示一个代理符号，可以作为事件表达式中的主体或客体
# group表示一个组，index表示在该组中的一个实体的索引，相当于使用AgentSymbol来代表这个实体
class AgentSymbol:
    """symbol to represent some agents"""

    def __init__(self, group, index):
        """ define a agent symbol, it can be the object or subject of EventNode

        group: group handle
            it is the return value of cfg.add_group()
        index: int or str
            int: a deterministic integer id
            str: can be 'all' or 'any', represents all or any agents in a group
        """
        self.group = group if group is not None else -1
        if index == 'any':
            self.index = -1
        elif index == 'all':
            self.index = -2
        else:
            # 如果assert后面的表达式为False，断言会触发一个AssertionError异常，
            # 并将字符串"index must be a deterministic int"作为异常的错误信息输出
            assert isinstance(self.index,
                              int), "index must be a deterministic int"
            self.index = index

    def __str__(self):
        return 'agent(%d,%d)' % (self.group, self.index)


# 场景的配置类
class Config:
    """configuration class of gridworld game"""

    def __init__(self):
        # 存储全局配置的字典
        self.config_dict = {}
        # 存储代理类型的字典
        self.agent_type_dict = {}
        self.groups = []
        self.reward_rules = []

    # 设置全局配置self.config_dict
    def set(self, args):
        """ set parameters of global configuration

        Parameters
        ----------
        args : dict
            key value pair of the configuration
        """
        for key in args:
            self.config_dict[key] = args[key]

    # 注册一个代理类型（创建一类智能体的属性比如运载机，而非单个智能体）
    def register_agent_type(self, name, attr):
        """ register an agent type

        Parameters
        ----------
        name : str
            name of the type (should be unique)
        attr: dict
            key value pair of the agent type
            see notes below to know the available attributes

        # attribute属性为字典类型
        Notes
        -----
        height: int, height of agent body
        width:  int, width of agent body
        speed:  float, maximum speed, i.e. the radius of move circle of the agent
        hp:     float, maximum health point of the agent
        view_range: gw.CircleRange or gw.SectorRange

        damage: float, attack damage
        step_recover: float, step recover of health point (can be negative)生命值的step回复量，可以为负值
        kill_supply: float, the hp gain when kill this type of agents 击杀这种单位时的hp收益

        step_reward: float, reward get in every step 
        kill_reward: float, reward gain when kill this type of agent 击杀这种单位的reward
        dead_penalty: float, reward get when dead 死亡的reward
        
        # 攻击处罚 （当单位展现attack属性时给的reward，为了防止单位攻击空白格子）
        attack_penalty: float, reward get when perform an attack (this is used to make agents do not attack blank grid)
        """
        # 创建的单位的存储格式为 name：attr的键值对，存储在self.agent_type_dict中
        # 每个单位都有一个单独的name，因此name重复时报错
        if name in self.agent_type_dict:
            raise Exception("type name %s already exists" % name)
        self.agent_type_dict[name] = attr
        return name

    # 将单位类型添加到组group中，返回的是索引（因为对group求长度，no就是agent_type的索引了）
    def add_group(self, agent_type):
        """ add a group to the configuration

        Returns
        -------
        group_handle : int
            a handle for the new added group
        """
        no = len(self.groups)
        self.groups.append(agent_type)
        return no

    # 添加reward规则
    def add_reward_rule(self, on, receiver, value, terminal=False):
        """ add a reward rule

        Some note:
        1. if the receiver is not a deterministic agent,
           it must be one of the agents involved in the triggering event

        Parameters
        ----------
        on: Expr
            a bool expression of the trigger event
        receiver:  (list of) AgentSymbol
            receiver of this reward rule
        value: (list of) float
            value to assign
        terminal: bool
            whether this event will terminate the game
        """
        """
        add_reward_rule 方法用于添加一个奖励规则。
        参数 on 是一个布尔表达式，表示触发事件。
        参数 receiver 是接收奖励的代理符号或其列表。
        参数 value 是分配的奖励值或其列表。
        参数 terminal 是一个布尔值，表示事件是否会终止游戏。
        """
        # 如果 receiver 和 value 不是列表，则将它们转换为列表
        if not (isinstance(receiver, tuple) or isinstance(receiver, list)):
            assert not (isinstance(value, tuple) or isinstance(value, tuple))
            receiver = [receiver]
            value = [value]
        # 如果 receiver 和 value 的长度不相等，则抛出异常
        if len(receiver) != len(value):
            raise Exception("the length of receiver and value should be equal")
        # 将奖励规则添加到 reward_rules 列表中
        self.reward_rules.append([on, receiver, value, terminal])


# 圆形范围
# 用于表示攻击或视野的范围
class CircleRange:

    def __init__(self, radius):
        """ define a circle range for attack or view

        Parameters
        ----------
        radius : float
        """
        self.radius = radius
        self.angle = 360

    def __str__(self):
        return 'circle(%g)' % self.radius


# 扇形范围
# 用于表示攻击或视野的范围
class SectorRange:

    def __init__(self, radius, angle):
        """ define a sector range for attack or view

        Parameters
        ----------
        radius : float
        angle :  float
            angle should be less than 180
        """
        self.radius = radius
        self.angle = angle
        if self.angle >= 180:
            raise Exception(
                "the angle of a sector should be smaller than 180 degree")

    def __str__(self):
        return 'sector(%g, %g)' % (self.radius, self.angle)
