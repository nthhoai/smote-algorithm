'''SMOTE(Dataset, Rate, K_neibor)
- parameters:
    + Dataset: (dataframe) Tập dữ liệu 
    + Rate: (int) tỷ lệ mong muốn số quan sát lớp thiểu số sau khi cân bằng so với ban đầu
    + K_neighbor: (int) k lân cận mong muốn
- output:
    + list (ma trận) chứa các mẫu synthetic
'''
import numpy as np
import pandas as pd
import random as rd
import math 


class SMOTE:
    num_att = 0 #số lượng biến
    index_of_minority = [] #lưu chỉ số của quan sát lớp thiểu số
    index_KNeighbor = [] #mảng lưu chỉ số của k index
    distance_df = pd.DataFrame()
    amount = 0 #số lượng mẫu synthetic cần sinh
    distance_array = [] #lưu khoảng cách của từng điểm thiểu số tới tất cả điểm còn lại
    #distance_index = pd.DataFrame() #lưu khoảng cách của từng điểm thiểu số tới tất cả điểm còn lại
    dataset = pd.DataFrame()
    x_dataframe = pd.DataFrame()
    y_dataframe = pd.DataFrame () #df mẫu thuộc lớp minority
    synthetic = [] #lưu mẫu synthetic
    flag = True

    def __init__(self, Dataset, Rate, K_neighbor):
        self.dataset = Dataset
        self.rate = Rate
        self.k_neighbor = K_neighbor

    def handleData(self):
        ''' XỬ LÝ DỮ LIỆU, trả về:
        + các index mẫu thuộc lớp thiểu số.
        + tập quan sát biến độc lập
        + Tập quan sát biến phụ thuộc tương ứng

        '''
        self.num_att = self.dataset.shape[1] #tổng số biến quan sát
        self.x_dataframe = self.dataset.iloc[: , :self.num_att-1]
        self.y_dataframe = self.dataset.iloc[: , -1:] #nhãn là cột cuối cùng của dataset 
        

        for i in range (len(self.y_dataframe)):
            temp = self.y_dataframe.iloc[i, 0]
            if temp == 0:
                self.index_of_minority.append(i)


    def calculateAmount(self):
        ''' Tính số mẫu synthetic cần tạo ra
       + Đầu vào: number_of_minority và rate 
       + Đầu ra: (rate/100) * number
        '''
        self.amount = int ((self.rate/100) * len(self.index_of_minority))
        if self.rate <= 100: #xét trường hợp tỷ lệ T nhập vào nhỏ hơn 100 --> chọn ngẫu nhiên trong lớp thiểu số tỷ lệ T tương ứng
            self.flag = False            
        return self.flag 

    def calculateDistance(self):
        '''
        [
        [0, 1.3, ...., 12.99] - m cột: khoảng cách đến m quan sát trong tập dữ liệu
        ....
        [1.2, 4.5, ....., 0] - n dòng: số lượng quan sát thuộc lớp thiểu số. 
        ]
        '''  
        for i in range (0, len(self.index_of_minority)): #tính khoảng cách từ 1 điểm thiểu số tới tất cả các điểm còn lại trong df
            distance = [] #lưu khoảng cách từ 1 điểm đc chọn đến n điểm còn lại
            point_or = self.index_of_minority[i] #lấy vị trí của điểm thiểu số 
            for j in range (len(self.dataset)): #j chạy từ 0 đến tổng số điểm trong df
                square_sum = 0
                for h in range (0, (self.num_att-1)): #h duyệt từng thuộc tính (chiều) để tính tổng bình phương, num_att -1: vì num_att là tổng các biến quan sát
                    square_sum = square_sum + math.pow ((self.x_dataframe.iloc[point_or][h] - self.x_dataframe.iloc[j][h]), 2) #cộng dồn bình phương chênh lệch 2 thuộc tính
                d = math.sqrt(square_sum) #căn bậc 2 tổng bình phương
                distance.append(d) #distance: mảng lưu khoảng cách từ điểm ban đầu tới n điểm còn lại
            self.distance_array.append(distance) #distance_array: ma trận nxm: n là số điểm quan sát, m là tổng số điểm
            del distance

    def selectKNeighbor(self):
        self.index_KNeighbor = np.argpartition(self.distance_array, self.k_neighbor+1, axis=1)[:,:self.k_neighbor+1]
        self.index_KNeighbor = self.index_KNeighbor.tolist()
        for i in range (len(self.index_KNeighbor)):
            if self.index_of_minority[i] in self.index_KNeighbor[i]:
                self.index_KNeighbor[i].remove(self.index_of_minority[i])

    def rateLess100(self):
        ''' HÀM NÀY DÙNG ĐỂ XỬ LÝ KHI rate < 100, chọn ngẫu nhiên số mẫu thiểu số'''
        while self.amount != 0:
            random = rd.randint (0,len(self.index_of_minority)-1)
            new_point = self.x_dataframe.iloc[random]
            new_point = new_point.values.tolist()
            self.synthetic.append(new_point) #thêm điểm mới vào list synthetic
            self.amount = self.amount - 1 #giảm số lượng mẫu cần sinh 1 đơn vị



    def rateHigher100(self):
        '''trong trường hợp rate  > 100'''
        while self.amount != 0:
            for i in range (0, len(self.index_of_minority)):
                #chọn ngẫu nhiên 1 lân cận trong số những lân cận gần nhất
                point_or = self.index_of_minority[i] #lấy ra vị trí của mẫu thiểu số đầu tiên
                random = rd.randint(0,(self.k_neighbor-1)) #chọn ngẫu nhiên lân cận thứ k
                ser = self.index_KNeighbor[i][random]  #vị trí của lân cận được chọn
                diff = self.x_dataframe.iloc[point_or] - self.x_dataframe.iloc[ser]  #độ chênh lệch giữa điểm k và điểm đang xét
                new_point = self.x_dataframe.iloc[point_or] + diff * round(rd.uniform(0,1), 3) #điểm mới sinh ra
                new_point = new_point.values.tolist()
                self.synthetic.append(new_point) #thêm điểm mới vào list synthetic
                self.amount = self.amount - 1 #giảm số lượng mẫu cần sinh 1 đơn vị

    def generateSmote(self):
        # tỷ lệ sinh mẫu > 100
        if self.flag == True:
            self.rateHigher100()
        #tỷ lệ sinh mẫu <= 100
        else:
            self.rateLess100()


    def smote (self):
        self.handleData()
        self.calculateAmount()
        self.calculateDistance()
        self.selectKNeighbor()
        self.generateSmote()


if __name__ == '__main__':


    Df_dataset = pd.read_csv("D:\PYTHON\processImbalancedData\data2\DLchuanhoa3.csv")


    a = SMOTE(Df_dataset,  50, 3)
    a.smote()
    df_synthetic = pd.DataFrame(a.synthetic)
    print (df_synthetic)
    




