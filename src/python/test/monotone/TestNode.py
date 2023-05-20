class TestNode:
    
    startValue = None
    endValue = None
    
    binBad = 0
    binTotal = 0
    
    cumBad = 0
    cumTotal = 0
    
    pre = None
    next = None
    
    def __init__(self, initStartValue, FirstTotal, FirstBad):
        self.binBad = FirstBad
        self.binTotal = FirstTotal
        self.cumTotal = FirstTotal
        self.cumBad = FirstBad 
        self.startValue = initStartValue
        self.endValue = initStartValue
    
    @property
    def mean(self) -> float:
        if self.cumTotal != 0 :
            return 1. * (self.cumBad / self.cumTotal)
        else :
            return 0
    
    @property
    def mergeTotal(self) -> int:
        return self.cumTotal
    
    @property
    def mergeBad(self) -> int:
        return self.cumBad
    
    def update(self, newTotal, newBad, newEnd) -> None :
        self.cumTotal += newTotal
        self.cumBad += newBad
        self.endValue = newEnd
        self.binBad += newBad
        self.binTotal += newTotal



# if __name__ == "__main__" :
    
#     def initTable(dataframe, var, default) :
#         return dataframe.groupby(var)[default].agg(["count", "sum"]).reset_index().rename(columns = {"count":"Total", "sum" :"Bad"})
    
#     df = pd.read_csv("/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv")
#     initTB = initTable(df, "Durationinmonth" ,"default")
    
#     root: TestNode = TestNode(0, 0, 0)
#     cur: TestNode = root
    
#     for i, row in enumerate(initTB.values):
#         _tmp = TestNode(row[0], row[1], row[2],) #define every node
#         cur.next = _tmp
#         _tmp.pre = cur
#         cur = cur.next
    
#     root = root.next # start from index = 1, `cur` compares with `cur.pre`
#     root.pre = None

#     # start comparison and merging
#     cur: TestNode = root
#     cur = cur.next
#     while cur is not None :
#         while cur.pre is not None and cur.mean <= cur.pre.mean:
#             cur.pre.update(cur.mergeTotal, cur.mergeBad, cur.endValue)
#             cur.pre.next = cur.next
#             if cur.next is not None:
#                 cur.next.pre = cur.pre
#             cur = cur.pre
#         cur = cur.next

#     # print result
#     cur = root
#     startValueList = []
#     endValueList = []
#     binTotalList = []
#     binBadList = []
#     meanList = []
#     while cur is not None:
#         startValueList.append(cur.startValue)
#         endValueList.append(cur.endValue)
#         binTotalList.append(cur.binTotal)
#         binBadList.append(cur.binBad)
#         meanList.append(cur.mean)
#         cur = cur.next
    
#     resDF = pd.DataFrame({"start":startValueList, "end": endValueList, "total": binTotalList, "bad": binBadList, "mean": meanList})
#     print(resDF)
