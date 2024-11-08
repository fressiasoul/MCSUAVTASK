import multiprocessing
import os

def run_file(filename):
    os.system('python {}'.format(filename))

if __name__ == '__main__':
    # 创建两个进程，分别运行两个Python文件
    #p1 = multiprocessing.Process(target=run_file, args=('run_pdqn_rsu.py',))
    #p2 = multiprocessing.Process(target=run_file, args=('run_pdqn_idol_vehicle.py',))
    p3 = multiprocessing.Process(target=run_file, args=('run_pdqn_rsu_idol_vehicle.py',))
    # p4 = multiprocessing.Process(target=run_file, args=('DQN/DQN.py',))
    # p5 = multiprocessing.Process(target=run_file, args=('DDPG/DDPG.py',))
    # p6 = multiprocessing.Process(target=run_file, args=('Random/Random.py',))
    # p7 = multiprocessing.Process(target=run_file, args=('Greedy/Greedy.py',))

    # 启动两个进程
    #p1.start()
    #p2.start()
    p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    # p7.start()
    # 等待两个进程结束
    #p1.join()
    #p2.join()
    p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    # p7.join()