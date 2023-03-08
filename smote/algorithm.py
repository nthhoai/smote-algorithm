'''SMOTE(Df_dataset, Df_minority, Rate, K_neibor)
- parameters:
    + Df_dataset: (dataframe) Tập dữ liệu 
    + Df_minority: (dataframe) Tập dữ liệu thuộc lớp minority 
    + Rate: (int) tỷ lệ mong muốn số quan sát lớp thiểu số sau khi cân bằng so với ban đầu
    + K_neibor: (int) k lân cận mong muốn
- output:
    + list (ma trận) chứa các mẫu synthetic
'''

import numpy as np
import pandas as pd
import random as rd
import math 
from scipy.spatial import distance


class SMOTE:
    number_of_minority = 0 #số lượng mẫu thuộc lớp minority
    number_sample = 0 #số lượng mẫu (thiểu số + đa số)
    num_att = 0 #số lượng thuộc tính
    amount = 0 #số lượng mẫu synthetic cần sinh
    distance_index = pd.DataFrame() #lưu khoảng cách của từng điểm thiểu số tới tất cả điểm còn lại
    point_k_index = pd.DataFrame() #lưu chỉ số của k lần cận gần nhất
    df_minority = pd.DataFrame () #df mẫu thuộc lớp minority
    df_dataset = pd.DataFrame() #df mẫu dataset
    synthetic = [] #lưu mẫu synthetic


    def __init__(self, Df_dataset, Df_minority, Rate, K_neibor):
        self.df_dataset = Df_dataset
        self.df_minority = Df_minority 
        self.rate = Rate
        self.k_neibor = K_neibor
        print ('Gọi hàm khởi tạo !')  

    def handleData(self):
        ''' XỬ LÝ DỮ LIỆU, trả về:
        + num_att: số lượng thuộc tính của dataset
        + number_of_minority: số lượng mẫu thuộc minority
        + number_sample: số lượng mẫu dataset       
        '''
        self.num_att = self.df_dataset.shape[1] 
        self.number_of_minority = len (self.df_minority)
        self.number_sample = len(self.df_dataset)
        #self.index_of_minority = self.df_minority.index 


    def calculateAmount(self):
        ''' Tính số mẫu synthetic cần tạo ra
       + Đầu vào: number_of_minority và rate 
       + Đầu ra: (rate/100) * number
        '''
        flag = False
        if self.rate < 100: #xét trường hợp tỷ lệ T nhập vào nhỏ hơn 100 --> chọn ngẫu nhiên trong lớp thiểu số tỷ lệ T tương ứng
            return True
        else: #ngược lại, sinh thêm lượng amount mẫu mới
            self.amount = int ((self.rate/100) * self.number_of_minority)


    def Tless100(self):
        ''' HÀM NÀY DÙNG ĐỂ XỬ LÝ KHI TỶ LỆ rate NHẬP VÀO < 100'''


    def calculateDistance(self):
        '''DÙNG scipy.spatial.distance.cdist
        '''  
        #chuyển dataframe sang np.array sao cho phù hợp với input hàm cdist 
        arr_mino = self.df_minority.to_numpy()
        arr_dataset = self.df_dataset.to_numpy()
        #tính khoảng cách từ điểm ban đầu tới các điểm còn lại
        #hàm cdist trả về ma trận n x m: 
        self.distance = np.round((distance.cdist(arr_mino, arr_dataset)), 3)

    def selectKNeighbor (self):
        ''' Chọn ra k lân cận gần nhất đối với từng mẫu lớp thiểu số, bằng cách:
        + chuyển đổi list chứa khoảng cách sang df
        + Sử dụng nsmallest trong pandas df
        + lưu index của k lân cận gần nhất.
        '''
        #chuyển đổi cấu trúc từ list sang dataframe
        distance_index = pd.DataFrame()
        for i in range (len(self.distance)):
            temp = pd.Series(self.distance[i])
            #mỗi cột là khoảng cách từ điểm ban đầu đến n điểm còn lại
            self.distance_index = pd.concat([self.distance_index, temp], axis=1) 

        #dùng function chọn index của k lân cận gần nhất        
        self.point_k_index = self.distance_index.apply(lambda s: pd.Series(s.nsmallest(self.k_neibor+1).index)) #index của k lân cận gần nhất, k= 3
        self.point_k_index = self.point_k_index.drop(0) #xóa đi dòng đầu tiên, vì dòng đầu khoảng cách = 0, trùng với điểm ban đầu được chọn tính khoảng cách      

    def generateSmote(self):
        '''trong trường hợp tỷ lệ T nhập vào > 100'''
        while self.amount != 0:
            for i in range (self.number_of_minority):
                #chọn ngẫu nhiên 1 lân cận trong số những lân cận gần nhất
                random = rd.randint(0,(self.k_neibor-1)) #chọn ngẫu nhiên lân cận thứ k
                ser = self.point_k_index.iloc[random, i]  #vị trí của lân cận được chọn: iloc[dòng random, cột i]
                diff = self.df_minority.iloc[i] - self.df_dataset.iloc[ser]  #độ chênh lệch giữa điểm k và điểm đang xét
                new_point = self.df_minority.iloc[i] + diff * round(rd.uniform(0,1), 3) #điểm mới sinh ra
                new_point = new_point.to_list()
                self.synthetic.append(new_point) #thêm điểm mới vào list synthetic
                self.amount = self.amount - 1 #giảm số lượng mẫu cần sinh 1 đơn vị

    def smote (self):
        self.handleData()
        self.calculateAmount()
        self.calculateDistance()
        self.selectKNeighbor()
        self.generateSmote()


if __name__ == '__main__':

    #xử lý sao cho phù hợp với đầu vào hàm SMOTE 
    minority = []
    Df_dataset = pd.read_csv("D:\PYTHON\processImbalancedData\data2\DLchuanhoa2.csv")
    for i in range (len(Df_dataset)):
            if Df_dataset['label'][i] == 0:
                minority.append(Df_dataset.loc[i])
    Df_minority = pd.DataFrame(minority)

    #khởi tạo đối tượng thuộc lớp smote và sử dụng hàm 
    a = SMOTE(Df_dataset, Df_minority, 300, 3)
    a.smote()
    print ("Số lượng mẫu minority ban đầu là: ", a.number_of_minority)
    print ("Số lượng mẫu synthetic cần sinh ra là: ", len(a.synthetic))
    
    #in các mẫu synthetic dưới dạng dataframe để dễ quan sát
    df = pd.DataFrame (a.synthetic)
    print (df)








        















    
