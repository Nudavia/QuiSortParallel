from mpi4py import MPI
import QuiSort as QS
import numpy as np
from time import time

Comm = MPI.COMM_WORLD
Rank = Comm.Get_rank()
Size = Comm.Get_size()


class QuiSortP:
    def __init__(self):
        self.dataset = [[] for _ in range(Size)]  # 根线程保存所有数据以及最后的结果
        self.localdata = []  # 个个线程的子数据
        self.subdata = []  # 没组的父线程从各子线程搜集的locadata，用来得到flag
        self.flag = 0  # 用来左右划分的比较对象

    def readFile(self, filepath):  # 读取文件
        infile = open(filepath)
        i = 0
        count = 0
        for line in infile:
            linelist = line.strip().split()
            for data in linelist:
                self.dataset[i].append(int(data))
                i = (i + 1) % Size
                count = count + 1
                if count > 500000:
                    break

    def partition(self):  # 类似快排的划分
        i = 0
        j = len(self.localdata) - 1
        while i < j:
            while i < j and self.localdata[j] >= self.flag:
                j = j - 1
            while i < j and self.localdata[i] <= self.flag:
                i = i + 1
            QS.swap(self.localdata, i, j)
        return j

    def swapForClassify(self, rankl, rankr):
        div = self.partition()  # div为划分边界点（属于小于的那一端）
        if div >= 0:
            step = int((rankr - rankl + 1) / 2)  # 和距离自己step的线程互换数据
            newdata = []
            if Rank <= int((rankr + rankl) / 2):  # 序号低的储存较小的元素
                dest = Rank + step
                # 这个地方可能是数据量太大不能一次性发过去所以先告诉对方你准备放多少个数据（确定接受端的迭代次数），再一个个发送
                # Comm.send(self.localdata[div + 1:], dest=dest)
                # newdata = Comm.recv(source=dest)
                Comm.send(len(self.localdata[div + 1:]), dest=dest)  # 把大的数的个数发过去，确定接受端迭代次数
                num = Comm.recv(source=dest)  # 获取对方要发送的数据个数
                for data in self.localdata[div:]:  # 一个个发送数据
                    Comm.send(data, dest=dest)
                for i in range(num):  # 接受数据
                    newdata.append(Comm.recv(source=dest))
                del self.localdata[div + 1:]  # 删除发送的数据
            else:  # 序号高的储存大的元素
                dest = Rank - step
                Comm.send(len(self.localdata[:div + 1]), dest=dest)
                num = Comm.recv(source=dest)
                # Comm.send(self.localdata[:div + 1], dest=dest)
                # newdata = Comm.recv(source=dest)
                for data in self.localdata[:div + 1]:
                    Comm.send(data, dest=dest)
                for i in range(num):
                    newdata.append(Comm.recv(source=dest))
                # print(len(newdata))
                del self.localdata[:div + 1]
            self.localdata.extend(newdata)  # 将接受的数据加入localdata

    def makeFlag(self):  # 将从子线程收集的数据通过三位取中发的到flag
        l = 0
        r = len(self.subdata) - 1
        if r >= l:
            m = int(l + (r - l) / 2)
            if self.subdata[r] > self.subdata[l]:
                if self.subdata[r] > self.subdata[m]:
                    select = m if self.subdata[m] > self.subdata[l] else l
                else:
                    select = r
            else:
                if self.subdata[l] > self.subdata[m]:
                    select = m if self.subdata[m] > self.subdata[r] else r
                else:
                    select = l
            self.flag = self.subdata[select]

    def classfy(self, rankl, rankr):  # 根据分组递归
        if rankl == rankr:  # 该组只剩一个线程直接排序
            QS.Sort(self.localdata)
            return
        if Rank != rankl:  # 其他线程向本组服线程发送数据
            Comm.send(self.localdata, dest=rankl)
        else:  # 父线程接受数据并产生flag再发给子线程
            self.subdata.extend(self.localdata)
            for rank in range(rankl + 1, rankr + 1):
                self.subdata.extend(Comm.recv(source=rank))
        ok = 0
        if Rank == rankl:
            if len(self.subdata) > 0:  # 必须有数据才能进行取中，避免极端情况
                for rank in range(rankl + 1, rankr + 1):
                    ok = 1
                    Comm.send(ok, dest=rank)  # 发送
                self.makeFlag()  # 获取flag
                for rank in range(rankl + 1, rankr + 1):
                    Comm.send(self.flag, dest=rank)
            else:
                for rank in range(rankl + 1, rankr + 1):
                    Comm.send(0, dest=rank)
        else:  # 如果ok就接收flag
            ok = Comm.recv(source=rankl)
            if ok == 1:
                self.flag = Comm.recv(source=rankl)
        if ok == 1:
            self.swapForClassify(rankl, rankr)  # 获取了flag之后就要根据flag划分并交换数据分组
            rankm = int((rankl + rankr) / 2)
            if Rank >= rankl and Rank <= rankm:  # 相应的进程分组才能进入相应递归（二分地进行递归）
                self.classfy(rankl, rankm)
            if Rank > rankm and Rank <= rankr:
                self.classfy(rankm + 1, rankr)

    def QSP(self, filepath):  # 并行快排
        if Rank == 0:
            self.readFile(filepath)  # 读取数据
        t1 = time()
        self.localdata = Comm.scatter(self.dataset, root=0)  # 散播
        self.classfy(0, Size - 1)  # 递归分组
        results = np.array(Comm.gather(self.localdata, root=0))  # 搜集各组排好序的数据，组与组之间由于分过类，也是有序的
        t2 = time()
        if Rank == 0:
            if Size == 1:  # 一维列表直接=
                self.dataset = results
            else:  # 多维列表extend
                self.dataset = []
                for result in results:
                    self.dataset.extend(result)
            print("耗时:" + str(t2 - t1))
            # print(self.dataset)


def main():
    # if Rank == 0:
    #     fo = open("data/dataset.txt", "w")
    #     for i in range(500000):
    #         fo.write(str(np.random.randint(0, 100000)))
    #         fo.write("    ")
    qsp = QuiSortP()
    qsp.QSP("data/dataset.txt")


if __name__ == '__main__':
    main()
