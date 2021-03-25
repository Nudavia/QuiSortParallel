from time import time
import numpy as np


# 将本地数据进行快排
def quiSort(list, l, r):
    if l < r:
        part = partition(list, l, r)
        quiSort(list, l, part - 1)
        quiSort(list, part + 1, r)


# 将区间二分(mid表示用不用三位取中)
def partition(list, l, r, mid=1):
    if mid == 1:
        makeFlag(list, l, r)
    i = l
    j = r
    while i < j:
        while (i < j and list[j] >= list[l]):
            j = j - 1
        while (i < j and list[i] <= list[l]):
            i = i + 1
        swap(list, i, j)
    swap(list, l, j)
    return j


# 根据本地数据选择flag放到第一个位置
def makeFlag(list, l, r):
    if r >= l:
        m = int(l + (r - l) / 2)
        if list[r] > list[l]:
            if list[r] > list[m]:
                select = m if list[m] > list[l] else l
            else:
                select = r
        else:
            if list[l] > list[m]:
                select = m if list[m] > list[r] else r
            else:
                select = l
        swap(list, l, select)


def swap(list, i, j):
    tmp = list[i]
    list[i] = list[j]
    list[j] = tmp


def Sort(list):
    quiSort(list, 0, len(list) - 1)


def main():
    dataset = []
    infile = open("data/dataset.txt")
    count=0
    for line in infile:
        linelist = line.strip().split()
        for data in linelist:
            dataset.append(int(data))
            count=count+1
            if count > 500000:
                break

    t1 = time()
    Sort(dataset)
    t2 = time()
    print("耗时:" + str(t2 - t1))
    #print(dataset)


if __name__ == '__main__':
    main()
