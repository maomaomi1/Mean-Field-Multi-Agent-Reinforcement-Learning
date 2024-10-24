import random
import math
import numpy as np

# 
# handles：不同group的句柄
# 作用为：在地图两侧生成智能体agents
def generate_map(env, map_size, handles):
    """ generate a map, which consists of two squares of agents"""
    width = height = map_size
    init_num = map_size * map_size * 0.04
    gap = 3

    leftID = random.randint(0, 1)
    rightID = 1 - leftID

    # left
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 - gap - side, width//2 - gap - side + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[leftID], method="custom", pos=pos)

    # right
    n = init_num
    side = int(math.sqrt(n)) * 2
    pos = []
    for x in range(width//2 + gap, width//2 + gap + side, 2):
        for y in range((height - side)//2, (height - side)//2 + side, 2):
            pos.append([x, y, 0])
    env.add_agents(handles[rightID], method="custom", pos=pos)


# 用于执行一个回合的训练或评估
def play(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """
    env：环境对象
    n_round：当前回合数
    map_size：地图大小。
    max_steps：每回合的最大步数。
    handles：环境中不同群体的句柄。
    models：用于控制代理的模型列表。
    print_every：打印信息的频率。
    eps：探索率（epsilon）。
    render：是否渲染环境。
    train：是否进行训练。
    """
    """play a ground and train"""
    
    # 重置环境并生成地图
    # reset包括：1、重置move和turn缓冲区 2、重置地图，并为地图的边界添加墙壁，清空相关内存，防止内存泄漏 3、重启渲染器和统计计数器
    # 4、清空所有的group和所有组内的智能体对象 5、重置奖励描述
    env.reset()
    # 在地图两侧生成两个group的agents
    generate_map(env, map_size, handles)

    # 初始化步数计数器，0步
    step_ct = 0
    # 训练是否完成
    done = False

    # 初始化用于存储状态、动作、ID、存活状态、奖励和数量（该group存活的单位数量）的列表，每个元素代表一个group的所有信息
    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    # 初始化损失和 Q 值评估的列表
    loss = [None for _ in range(n_group)]
    eval_q = [None for _ in range(n_group)]
    # 获取各个group的动作空间大小
    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    # 输出示例：[*] ROUND #5, EPS: 0.12 NUMBER: 100
    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    
    # 初始化每个组的平均reward和总reward
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    # 前一个动作的概率的列表（大小为动作空间大小，两个group）
    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    # 游戏没有结束，循环max_steps个仿真步
    while not done and step_ct < max_steps:
        # take actions for every model
        # 获取每个group的所有智能体的观测数据
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            # 组中所有单位的id，numpy数组
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            # np.tile(A, reps)为重复A数组reps次数
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            # 每个模型根据状态和 “前一个动作的概率” 选择动作
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        # 将选择的动作设置到环境中
        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        # 进行一个仿真步（若一方所有单位死亡，仿真结束）
        done = env.step()

        # 获取各个group的奖励和存活状态
        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        # 创建缓存区，存储状态、动作、奖励等所有参数
        buffer = {
            'state': state[0], 'acts': acts[0], 'rewards': rewards[0],
            'alives': alives[0], 'ids': ids[0]
        }
        buffer['prob'] = former_act_prob[0]

        # 更新“前一个动作的概率”
        for i in range(n_group):
            # map内将acts[i] 中的每个动作转换为 one-hot 向量
            """如[[1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]]表示5个智能体的动作分别为[0,1,0,2,1]，然后计算平均值[0.4, 0.4, 0.2]
            """
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # 如果处在训练状态，更新各个智能体的缓存区
        if train:
            models[0].flush_buffer(**buffer)

        # stat info更新统计信息
        # 获取种群的代理数量
        nums = [env.get_num(handle) for handle in handles]

        # 计算每个group的总reward和平均reward
        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            # 记录历史奖励（mean_rewards和total_rewards记录每一步的奖励）
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        # 环境渲染
        if render:
            env.render()

        # clear dead agents
        env.clear_dead()
        
        # np.round是将数组中的每个元素保留小数点后6位
        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        # 仿真步+1
        step_ct += 1

        # 每print_every个仿真步，打印步数和reward以及group内智能体数量信息
        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    # 一个回合结束后，模型进行一次训练，更新参数
    # 通过积累多个时间步的数据，批量更新允许使用矩阵运算，提高计算效率
    if train:
        models[0].train()

    for i in range(n_group):
        # mean_rewards[i]原来记录的是该次仿真每个step的reward，现一个回合结束后，计算所有step的平均奖励，覆盖原位置
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        # 计算所有step的和奖励的和，覆盖原位置
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards


def battle(env, n_round, map_size, max_steps, handles, models, print_every, eps=1.0, render=False, train=False):
    """play a ground and train"""
    env.reset()
    generate_map(env, map_size, handles)

    step_ct = 0
    done = False

    n_group = len(handles)
    state = [None for _ in range(n_group)]
    acts = [None for _ in range(n_group)]
    ids = [None for _ in range(n_group)]

    alives = [None for _ in range(n_group)]
    rewards = [None for _ in range(n_group)]
    nums = [env.get_num(handle) for handle in handles]
    max_nums = nums.copy()

    n_action = [env.get_action_space(handles[0])[0], env.get_action_space(handles[1])[0]]

    print("\n\n[*] ROUND #{0}, EPS: {1:.2f} NUMBER: {2}".format(n_round, eps, nums))
    mean_rewards = [[] for _ in range(n_group)]
    total_rewards = [[] for _ in range(n_group)]

    former_act_prob = [np.zeros((1, env.get_action_space(handles[0])[0])), np.zeros((1, env.get_action_space(handles[1])[0]))]

    while not done and step_ct < max_steps:
        # take actions for every model
        for i in range(n_group):
            state[i] = list(env.get_observation(handles[i]))
            ids[i] = env.get_agent_id(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.tile(former_act_prob[i], (len(state[i][0]), 1))
            acts[i] = models[i].act(state=state[i], prob=former_act_prob[i], eps=eps)

        for i in range(n_group):
            env.set_action(handles[i], acts[i])

        # simulate one step
        done = env.step()

        for i in range(n_group):
            rewards[i] = env.get_reward(handles[i])
            alives[i] = env.get_alive(handles[i])

        for i in range(n_group):
            former_act_prob[i] = np.mean(list(map(lambda x: np.eye(n_action[i])[x], acts[i])), axis=0, keepdims=True)

        # stat info
        nums = [env.get_num(handle) for handle in handles]

        for i in range(n_group):
            sum_reward = sum(rewards[i])
            rewards[i] = sum_reward / nums[i]
            mean_rewards[i].append(rewards[i])
            total_rewards[i].append(sum_reward)

        if render:
            env.render()

        # clear dead agents
        env.clear_dead()

        info = {"Ave-Reward": np.round(rewards, decimals=6), "NUM": nums}

        step_ct += 1

        if step_ct % print_every == 0:
            print("> step #{}, info: {}".format(step_ct, info))

    for i in range(n_group):
        mean_rewards[i] = sum(mean_rewards[i]) / len(mean_rewards[i])
        total_rewards[i] = sum(total_rewards[i])

    return max_nums, nums, mean_rewards, total_rewards
