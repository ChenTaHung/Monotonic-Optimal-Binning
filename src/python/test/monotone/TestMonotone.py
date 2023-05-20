class Node:
    __merge = 1
    __sum = 0
    pre = None
    next = None

    def __init__(self, init_sum):
        self.__sum = init_sum

    @property
    def sum(self) -> int:
        return self.__sum
    
    @property
    def merge(self) -> int:
        return self.__merge
    
    @property
    def mean(self) -> float:
        return 1. * self.__sum / self.__merge
    
    def update(self, merge, sum) -> None:
        self.__merge += merge
        self.__sum += sum

def main():
    # Inits double link list
    root: Node = Node(0)
    cur: Node = root
    with open('/Users/chentahung/Downloads/in.txt', 'r') as f:
        line = f.readline()
        for data in line.split(' '):
            tmp: Node = Node(int(data))
            cur.next = tmp
            tmp.pre = cur
            cur = cur.next
    root = root.next
    root.pre = None

    # Starts comparison and merging by the 2nd element
    cur: Node = root
    cur = cur.next
    while cur is not None:
        while cur.pre is not None and cur.mean <= cur.pre.mean:
            cur.pre.update(cur.merge, cur.sum)
            cur.pre.next = cur.next
            if cur.next is not None:
                cur.next.pre = cur.pre
            cur = cur.pre
        cur = cur.next

    # print result
    cur = root
    while cur is not None:
        print(cur.mean)
        cur = cur.next

main()