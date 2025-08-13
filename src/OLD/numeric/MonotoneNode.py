import numpy as np

class MonotoneNode:
    
    pre = None
    next = None
    
    def __init__(self, Value, FirstTotal, FirstBad, FirstStd):
        self.cumStd = FirstStd
        self.cumTotal = FirstTotal
        self.cumBad = FirstBad 
        self.startValue = Value
        self.endValue = Value
    
    @property
    def mean(self) -> float:
        if self.cumTotal != 0 :
            return 1. * (self.cumBad / self.cumTotal)
        else :
            return 0
    @property
    def mergeStd(self) -> float :
        return self.cumStd
    
    @property
    def mergeTotal(self) -> int:
        return self.cumTotal
    
    @property
    def mergeBad(self) -> int:
        return self.cumBad
    
    def update(self, newTotal, newBad, newStd, newEnd) -> None :
        
        if (self.cumTotal + newTotal) == 2 :
            self.cumStd = np.std(np.array(self.mean, (newBad/newTotal))) # newMean = newBad / newTotal
        else :
            self.cumStd = np.sqrt(((self.cumTotal * (self.cumStd ** 2)) + (newTotal * (newStd ** 2))) / (self.cumTotal + newTotal - 1))
        
        self.cumTotal += newTotal
        self.cumBad += newBad
        self.endValue = newEnd
            



# if __name__ == '__main__' :
#     import pandas as pd
    
#     def initTable(dataframe, var, default) :
#         return dataframe.groupby(var)[default].agg(['count', 'sum', 'std']).reset_index().rename(columns = {'count':'Total', 'sum' :'Bad'}).fillna(0)
    
#     df = pd.read_csv('/Users/chentahung/Desktop/git/mob-py/data/german_data_credit_cat.csv')
#     df['default'] = df['default'] - 1
#     initTB = initTable(df, 'Durationinmonth' ,'default')
    
#     root: MonotoneNode = MonotoneNode(0,0,0,0)
#     cur: MonotoneNode = root
    
#     for row in initTB.values:
#         _tmp = MonotoneNode(Value = row[0], FirstTotal = row[1], FirstBad = row[2], FirstStd = row[3]) #define every node
#         cur.next = _tmp
#         _tmp.pre = cur
#         cur = cur.next
    
#     root = root.next # start from index = 1, `cur` compares with `cur.pre`
#     root.pre = None

#     # start comparison and merging
#     cur: MonotoneNode = root
#     cur = cur.next
#     while cur is not None :
#         while cur.pre is not None and cur.mean <= cur.pre.mean:
#             cur.pre.update(cur.mergeTotal, cur.mergeBad, cur.mergeStd, cur.endValue)
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
#     stdList = []
#     while cur is not None:
#         startValueList.append(cur.startValue)
#         endValueList.append(cur.endValue)
#         binTotalList.append(cur.cumTotal)
#         binBadList.append(cur.cumBad)
#         stdList.append(cur.cumStd)
#         meanList.append(cur.mean)
#         cur = cur.next
    
#     resDF = pd.DataFrame({'start':startValueList, 'end': endValueList, 'total': binTotalList, 'bad': binBadList, 'mean': meanList, 'std': stdList})
#     print(resDF)
