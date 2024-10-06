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
    parser.add_argument('--algo', type=str, choices={'ac', 'mfac', 'mfq', 'il'}, help='choose an algorithm from the preset', required=True)
    # 设置保存间隔
    parser.add_argument('--save_every', type=int, default=10, help='decide the self-play update interval')
    # 设置Q学习的更新间隔（可选）
    parser.add_argument('--update_every', type=int, default=5, help='decide the udpate interval for q-learning, optional')
    # 训练轮数train round（指一个完整周期，一个train round包括多个episode）
    parser.add_argument('--n_round', type=int, default=2000, help='set the trainning round')
    # 是否渲染（render）
    parser.add_argument('--render', action='store_true', help='render or not (if true, will render every save)')
    # 地图大小
    parser.add_argument('--map_size', type=int, default=40, help='set the size of map')  # then the amount of agents is 64
    # max step（每个智能体在一个回合episode中可以执行的最大动作步数）
    parser.add_argument('--max_steps', type=int, default=400, help='set the max steps')

    args = parser.parse_args()

    # Initialize the environment
    env = magent.GridWorld('battle', map_size=args.map_size)
    # 设置渲染文件的保存目录（gridworld.py中的方法）（os.path.join将几个路径拼接起来）
    env.set_render_dir(os.path.join(BASE_DIR, 'examples/battle_model', 'build/render'))
    # 获取智能体的句柄（源代码中self.group_handles.append(handle)应该是指group中的所有单位）
    handles = env.get_handles()

    # TensorFlow配置
    # allow_soft_placement=True启用软设备放置：如果GPU不适配，会自动选择CPU计算
    # log_device_placement：关闭设备放置的日志输出，避免不必要的信息干扰调试输出
    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # 控制 TensorFlow 如何分配 GPU 内存，表示 TensorFlow 会根据程序运行的需要，逐渐增加 GPU 内存的分配，而不是一次性占用所有的 GPU 内存
    tf_config.gpu_options.allow_growth = True

    # 设置日志和模型的保存日志
    log_dir = os.path.join(BASE_DIR,'data/tmp'.format(args.algo))
    # {}表示占位符，与format搭配将args.algo的值填入其中
    model_dir = os.path.join(BASE_DIR, 'data/models/{}'.format(args.algo))

    if args.algo in ['mfq', 'mfac']:
        use_mf = True
    else:
        use_mf = False

    start_from = 0

    # 根据配置的tf_config创建一个TF的session（会话），用于运行TensorFlow的计算图。会话管理了计算图中的所有操作和变量的执行。
    sess = tf.Session(config=tf_config)
    # 创建两个
    models = [spawn_ai(args.algo, sess, env, handles[0], args.algo + '-me', args.max_steps), spawn_ai(args.algo, sess, env, handles[1], args.algo + '-opponent', args.max_steps)]
    sess.run(tf.global_variables_initializer())
    runner = tools.Runner(sess, env, handles, args.map_size, args.max_steps, models, play,
                            render_every=args.save_every if args.render else 0, save_every=args.save_every, tau=0.01, log_name=args.algo,
                            log_dir=log_dir, model_dir=model_dir, train=True)

    for k in range(start_from, start_from + args.n_round):
        eps = linear_decay(k, [0, int(args.n_round * 0.8), args.n_round], [1, 0.2, 0.1])
        runner.run(eps, k)
