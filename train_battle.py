"""Self Play
"""

import argparse
import os
import tensorflow as tf
import numpy as np
import magent

from examples.battle_model.algo import spawn_ai
from examples.battle_model.algo import tools
from examples.battle_model.senario_battle import play

# 返回当前脚本所在的目录，赋值给basedir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 定义线性衰减函数，根据训练的轮数线性地衰减参数值
def linear_decay(epoch, x, y):
    """
    epoch: 当前的时间点（或迭代次数），用于决定返回哪个值。
    x: 一个列表，表示时间点的集合，通常是整数或浮点数，按升序排列。
    y: 一个列表，表示与时间点 x 对应的值，通常是浮点数，表示在这些时间点的目标值。
    其实就是插值
    """
    min_v, max_v = y[0], y[-1]
    start, end = x[0], x[-1]

    if epoch == start:
        return min_v

    eps = min_v

    for i, x_i in enumerate(x):
        if epoch <= x_i:
            interval = (y[i] - y[i - 1]) / (x_i - x[i - 1])
            eps = interval * (epoch - x[i - 1]) + y[i - 1]
            break

    return eps


if __name__ == '__main__':
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser()
    # 使用的训练算法
    parser.add_argument('--algo',
                        type=str,
                        choices={'ac', 'mfac', 'mfq', 'il'},
                        help='choose an algorithm from the preset',
                        required=True)
    # 设置保存间隔
    parser.add_argument('--save_every',
                        type=int,
                        default=10,
                        help='decide the self-play update interval')
    # 设置Q学习的更新间隔（可选）
    parser.add_argument(
        '--update_every',
        type=int,
        default=5,
        help='decide the udpate interval for q-learning, optional')
    # 训练轮数train round（指一个完整周期，一个train round包括多个episode）
    parser.add_argument('--n_round',
                        type=int,
                        default=2000,
                        help='set the trainning round')
    # 是否渲染（render）
    parser.add_argument('--render',
                        action='store_true',
                        help='render or not (if true, will render every save)')
    # 地图大小
    parser.add_argument(
        '--map_size', type=int, default=40,
        help='set the size of map')  # then the amount of agents is 64
    # max step（每个智能体在一个回合episode中可以执行的最大动作步数）
    parser.add_argument('--max_steps',
                        type=int,
                        default=400,
                        help='set the max steps')

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    # 设置渲染文件的保存目录（gridworld.py中的方法）（os.path.join将几个路径拼接起来）
    env.set_render_dir(
        os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    # 两个group的句柄
    handles = env.get_handles()

    # TensorFlow配置
    # allow_soft_placement=True启用软设备放置：如果GPU不适配，会自动选择CPU计算
    # log_device_placement：关闭设备放置的日志输出，避免不必要的信息干扰调试输出
    tf_config = tf.ConfigProto(allow_soft_placement=True,
                               log_device_placement=False)
    # 控制 TensorFlow 如何分配 GPU 内存，表示 TensorFlow 会根据程序运行的需要，逐渐增加 GPU 内存的分配，而不是一次性占用所有的 GPU 内存
    tf_config.gpu_options.allow_growth = True

    # 设置日志和模型的保存日志
    log_dir = os.path.join(BASE_DIR, 'data/tmp'.format(args.algo))
    # {}表示占位符，与format搭配将args.algo的值填入其中
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    if args.algo in ['mfq', 'mfac']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    # 根据配置的tf_config创建一个TF的session（会话），用于运行TensorFlow的计算图。会话管理了计算图中的所有操作和变量的执行。
    sess = tf.Session(config=tf_config)
    # 创建两个模型（代表正反方两个group：handles[0]和handles[1]），在这里是MFQ(base.ValueNet)模型
    # 两个模型共享一个session计算图，可以共享计算资源和内存，提高效率
    models = [
        spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me',
                 args.max_steps),
        spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent',
                 args.max_steps)
    ]

    # 初始化全局变量（在base.py中通过 tf.Variable 创建的变量
    # 而占位符 tf.placeholder 用于在运行时传递数据，不需要初始化
    sess.run(tf.global_variables_initializer())

    # 创建runner对象，Runner 对象负责运行训练或测试过程，管理整个训练流程
    # from examples.battle_model.senario_battle import play
    # play输入为运行一次训练的函数handle
    runner = tools.Runner(sess,
                          env,
                          handles,
                          args.map_size,
                          args.max_steps,
                          models,
                          play,
                          render_every=args.save_every if args.render else 0,
                          save_every=args.save_every,
                          tau=0.01,
                          log_name=args.algo,
                          log_dir=log_dir,
                          model_dir=model_dir,
                          train=True)

    # 实现训练循环
    for k in range(start_from, start_from + args.n_round):
        # 计算当前轮次的探索率 eps，根据线性衰减策略。
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round],
                           [1, 0.2, 0.1])
        # 进行一次训练或测试
        runner.run(eps, k)
