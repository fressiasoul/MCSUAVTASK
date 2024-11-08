import pickle
import random

import numpy as np
import pandas as pd

import gym
from gym import spaces

# 1.use_idol_vehicle=True, use_rsu=True
# 2.use_idol_vehicle=True, use_rsu=False
# 3.use_idol_vehicle=True, use_rsu=True

# np.random.seed(1)

class TaskOffloadingEnv(gym.Env):

    def __init__(self,
                 n_ue=5,
                 n_en=5,
                 n_task=5,
                 use_en=True,
                 use_bs=True,
                 map_size=200):
        """
        :param n_ue    用户的数量 K
        :param n_bs    基站数量      N
        :param n_en    闲置用户的数量 H
        :n_task        总任务数量    M
        :use_en        是否卸载到闲置用户，默认为真
        :use_bs        是否卸载到基站，默认为真
        :map_size      地图大小，默认为100x100
        """
        super(TaskOffloadingEnv, self).__init__()

        self.use_en = use_en
        self.use_bs = use_bs

        # 定义闲置工人和无人机和任务的数量
        self.n_ue = self.K = n_ue
        self.n_en = self.H = n_en
        self.n_task_per_ue = self.M = n_task

        # 定义地图的边长
        self.map_size = map_size

        # 定义固定的任务属性
        self.tasks_prop = np.array([
            [50, 50, 20, 1, 100],  # [x, y, 截至时间, 优先级, 基础报酬]
            [75, 75, 20, 3, 150],
            [100, 50, 20, 5, 250],
            [125, 35, 20, 7, 200],
            [180, 90, 20, 9, 150]
        ])

        # 定义固定的工人属性
        self.position_ue = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.pay_ue = np.array([
            [11],
            [12],
            [13],
            [14],
            [15]
        ])
        self.ue_prop = np.hstack((self.position_ue, self.pay_ue))

        # 定义固定的无人机属性
        self.position_en = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0]
        ])
        self.pay_en = np.array([
            [20],
            [20],
            [20],
            [20],
            [20]
        ])
        self.en_prop = np.hstack((self.position_en, self.pay_en))


        # 定义当前执行任务的索引号
        self.cur_task = 0
        self.now=0

        # 维护工人和无人机的计算资源暂用情况  初始化为1（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        # 状态空间，四行分别是：
        # 任务属性 [x, y, 截至时间, 优先级, 基础报酬]
        # 当前执行任务的用户和无人机索引号
        # 工人和无人机的属性 [x, y, 报酬/无人机能耗]
        self.observation_space = 56
        self.observation_space = spaces.Discrete(self.observation_space)

        self.action_space = 1 + self.n_en + self.n_ue + 2
        # NOTE: 接入 openai gym 接口
        self.action_space = spaces.Tuple((
            spaces.Discrete(self.action_space - 2),
            spaces.Tuple(
                tuple(spaces.Box(low=np.array([0, 0]), high=np.array([1, 1]), dtype=np.float32)
                      for _ in range(self.action_space - 2))
            )
        ))


    def reset(self):
        # np.random.seed(1)
        """
        随机初始化
        """


        # 维护闲置车辆和路边单元设备的计算资源暂用情况（[0,1]）
        self.resource_ue = np.ones(self.n_ue)
        self.resource_en = np.ones(self.n_en)

        # 记录工人和无人机能否卸载[0,1] 设置1 or 0 ？
        self.mask_ue = np.ones(shape=self.n_ue)
        self.mask_en = np.ones(shape=self.n_en)

        # 定义当前执行任务的索引号
        self.cur_task = 0
        self.now = 0

        state = self._state()

        self.cur_task = 0

        return state
    # 0 不能卸载到自身 1 可以
    def get_mask_action(self):
        return np.concatenate([np.array([1]), self.mask_ue, self.mask_en], axis=-1)

    def step(self, action):
        reward = self._reward(action)
        reward = np.clip(reward, -1000, np.inf)

        self._update_state(action)

        state = self._state()
        done = self.now >= len(self.tasks_prop)
        return state, reward, done, {}

    def _state(self):
        return np.concatenate((
            self.tasks_prop[self.cur_task],
            [self.cur_task],
            self.resource_ue,
            self.resource_en,
            self.mask_ue,
            self.mask_en,
            self.position_ue.flatten(),
            self.position_en.flatten(),
            self.pay_ue.flatten(),
            self.pay_en.flatten()
        ))

    def _reward_ue(self, action):
        x=action[0]
        if (x >= 10):
            x = x - 10
        task_prop = self.tasks_prop[self.cur_task]
        ue_prop = self.ue_prop[x]
        speed = action[1][action[0]][0] + 2
        d0 = np.linalg.norm(self.position_ue[x] - self.tasks_prop[self.cur_task][:2])
        t_ij = d0/speed
        get = task_prop[3] *10+ task_prop[4]+(task_prop[2]-t_ij)*10
        pay = ue_prop[2]
        reward = get - pay
        # 增加基于任务完成进度的奖励
        return reward

    def _reward_en(self, action):
        x = action[0]
        if (x >= 10):
            x = x - 10
        task_prop = self.tasks_prop[self.cur_task]
        en_prop = self.en_prop[x - 5]
        speed = action[1][action[0]][1] + 10  # 寻找合适函数转换
        d0 = np.linalg.norm(self.position_en[x - 5] - self.tasks_prop[self.cur_task][:2])
        t_ij = d0/speed
        get = task_prop[3]*10 + task_prop[4]*1.25+(task_prop[2]-t_ij)*10
        pay = en_prop[2]*t_ij
        reward = get - pay
        # 增加基于任务完成进度的奖励
        return reward

    def _reward(self, action):
        print(action)
        action_x = action[0]
        if action_x>=10:
            action_x=action_x-10
        if action_x < self.n_ue:
            return self._reward_ue(action)-3
        else:
            return self._reward_en(action)-1.5

    def _update_state(self, action):
        action = action[0]
        if action>=10:
            action=action-10
        if action < self.n_ue:
            self.resource_ue[action] = 0
            self.mask_ue[action] = 0
        else:
            self.resource_en[action - self.n_ue] = 0
            self.mask_en[action - self.n_ue] = 0
        self.cur_task += 1
        if self.cur_task >= len(self.tasks_prop):
            self.cur_task = len(self.tasks_prop) - 1
            self.now = len(self.tasks_prop)


class TaskOffloadingEnvActionWrapper(gym.ActionWrapper):
    """
    Changes the format of the parameterised action space to conform to that of Goal-v0 and Platform-v0
    """
    def __init__(self, env):
        super(TaskOffloadingEnvActionWrapper, self).__init__(env)
        old_as = env.action_space
        self.num_actions = old_as.spaces[0].n
        num_actions = old_as.spaces[0].n
        print(self.num_actions)
        self.action_space = gym.spaces.Tuple((
            old_as.spaces[0],  # actions
            *(gym.spaces.Box(np.float32(old_as.spaces[1].spaces[i].low), np.float32(old_as.spaces[1].spaces[i].high),
                             dtype=np.float32)
              for i in range(0, num_actions))
        ))

    def action(self, action):
        return action


if __name__ == "__main__":
    env = TaskOffloadingEnv()

    state = env.reset()

    for _ in range(100):
        a = np.random.randint(0, 10, size=1)
        bandwidth = np.random.random(1)
        compute = np.random.random(1)

        state_, reward, done, info = env.step(np.array([[0, 1, 2], [[0.4, 0.3], [0.4, 0.3], [0.4, 0.3]]], dtype=int))
        if done: break
        print(state_)
