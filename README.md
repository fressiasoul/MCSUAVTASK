# 使用文档

## 参考代码

![https://github.com/cycraig/MP-DQN](参考仓库代码)

## 文件说明

```
agents							# 存放智能体（P-DQN）算法
data							# 训练过程中的奖励曲线（可以直接绘图）
log-ok							# SummaryWriter的日志文件夹（可以取数据用于绘图）
model							# 二元分类模型文件
results-ok						# 训练好的模型文件（可以直接使用）
env.py							# 环境文件
main.py							# 同时训练三个模型
run_pdqn_idol_vehicle.py		# 仅仅卸载到闲置车辆
run_pdqn_rsu.py					# 仅仅卸载到RSU
run_pdqn_rsu_idol_vehicle.py	# 同时卸载到闲置车辆和RSU
```

## 方法说明

本次代码实现主要为env和run_文件，下面以run_pdqn_idol_vehicle.py为例进行说明。

### env.py

环境这部分主要是参考了openai的gym库，主要包含了如下几个方法

```
reset()                         # 重置环境
step(action)                    # 执行动作

_state()                        # 获取当前状态
_reward()                       # 获取当前奖励
_reward_self()                  # 获取卸载到自身的奖励
_reward_rsu()                   # 获取卸载到RSU的奖励
_reward_idol_vehicle()          # 获取卸载到闲置车辆的奖励
_update_state()                 # 更新环境状态

_load_model()                   # 加载分类编码器
_task_type()                    # 获取任务类型（使用编码器）
_generate_task_type()           # 生成任务类型（从文件读取属性）
```

此外env.py文件还包含了TaskOffloadingEnvActionWrapper类，其主要是对原始环境包裹，使得包裹后的状态空间和动作空间能够对接pdqn算法。

### run_pdqn_idol_vehicle.py

这三个run开头的代码文件，主要是对env.py中的环境进行调用，同时调用agents文件夹中的算法进行训练。

### agents文件夹

agents文件夹中包含了P-DQN算法的实现，主要使用到了如下几个文件

```
memory/memory.py                # 经验回放池
utils/noise.py                  # 实现了噪声，主要是OU噪声
agent.py                        # 智能体接口（定义了方法）
pdqn.py                         # 实现了具体的P-DQN算法
```

P-DQN算法的代码主要是为pdqn.py文件，其主要包含了run（用于训练）和evaluate（用于测试）两个方法