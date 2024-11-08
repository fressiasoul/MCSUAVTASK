import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

data_frame_all = pd.read_csv('run-2023-05-01 20_03_11.459319-tag-episode_reward.csv')
data_frame_no_idol = pd.read_csv("run-2023-05-01 20_03_11.454335-tag-episode_reward.csv")
data_frame_no_rsu = pd.read_csv('run-2023-05-01 20_03_11.480380-tag-episode_reward.csv')

x = data_frame_all['Step'][:500]

y1 = data_frame_all['Value'][:500]
y2 = data_frame_no_idol['Value'][:500]
y3 = data_frame_no_rsu['Value'][:500]

#plt.figure(figsize=(600,400),dpi=100)

fig, ax = plt.subplots()
#.plot(x, y)
ax.plot(x,y1,label = 'all')
ax.plot(x,y2,label = 'rsu')
ax.plot(x,y3,label = 'idol_vehicle')

# 设置坐标轴范围
# ax.set_xlim(0, 800)
# ax.set_ylim(-350, 200)

# 在指定的范围内放大坐标轴
axins =ax.inset_axes([0.58, 0.4, 0.4, 0.4])  # (left, bottom, width, height)
#.plot(x, y)
axins.plot(x,y1,label = 'all')
axins.plot(x,y2,label = 'rsu')
axins.plot(x,y3,label = 'idol_vehicle')
axins.set_xlim(400, 500)
axins.set_ylim(-400, 200)

mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.legend()
plt.show()
