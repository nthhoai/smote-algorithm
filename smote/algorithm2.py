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

    amount = 0 #số lượng mẫu synthetic cần sinh
    distance_array = [] 
    #distance_index = pd.DataFrame() #lưu khoảng cách của từng điểm thiểu số tới tất cả điểm còn lại

    dataset = pd.DataFrame()
    x_dataframe = pd.DataFrame()
    y_dataframe = pd.DataFrame () #df mẫu thuộc lớp minority
    synthetic = [] #lưu mẫu synthetic


    def __init__(self, Dataset, Rate, K_neighbor):
        self.dataset = Dataset
        self.rate = Rate
        self.k_neighbor = K_neighbor
        print ('Gọi hàm khởi tạo !')  

    def handleData(self):
        ''' XỬ LÝ DỮ LIỆU, trả về:
        + các index mẫu thuộc lớp thiểu số.
        + tập quan sát biến độc lập
        + Tập quan sát biến phụ thuộc tương ứng

        '''
        num_att = self.dataset.shape[1] #tổng số biến quan sát
        self.x_dataframe = self.dataset.iloc[: , :num_att-1]
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
        flag = False
        if self.rate < 100: #xét trường hợp tỷ lệ T nhập vào nhỏ hơn 100 --> chọn ngẫu nhiên trong lớp thiểu số tỷ lệ T tương ứng
            return True
        else: #ngược lại, sinh thêm lượng amount mẫu mới
            self.amount = int ((self.rate/100) * len(self.index_of_minority))


    def Tless100(self):
        ''' HÀM NÀY DÙNG ĐỂ XỬ LÝ KHI TỶ LỆ rate NHẬP VÀO < 100'''


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
            self.point_or = self.index_of_minority[i] #lấy vị trí của điểm thiểu số 
            for j in range (0, len(self.dataset)): #j chạy từ 0 đến tổng số điểm trong df
                square_sum = 0
                for h in range (0, self.num_att-1): #h duyệt từng thuộc tính (chiều) để tính tổng bình phương, num_att -1: vì num_att là tổng các biến quan sát
                    square_sum = square_sum + math.pow ((self.dataset.iloc[self.point_or][h] - self.dataset.iloc[j][h]), 2) #cộng dồn bình phương chênh lệch 2 thuộc tính
                d = math.sqrt(square_sum) #căn bậc 2 tổng bình phương
                distance.append(d) #distance: mảng lưu khoảng cách từ điểm ban đầu tới n điểm còn lại
            self.distance_array.append(distance) #distance_array: ma trận nxm: n là số điểm quan sát, m là tổng số điểm
            del distance

    def selectKNeighbor (self):
        ''' TÌM LẦN LƯỢT: TÌM LÂN CẬN THỨ NHẤT SAU ĐÓ XÓA NÓ ĐI ĐỂ THỰC HIỆN LẦN TÌM LÂN CẬN TIẾP THEO
        '''
        for i in range (0, len(self.index_of_minority)): #duyệt các mẫu trong lớp thiểu số
            distance_temp = self.distance_array[i] #truy cập vào mảng lưu khoảng cách của điểm thiểu số thứ i (1)
            index_KNeighbor_temp = [] #dùng để lưu CHỈ SỐ của k lân cận
            for j in range (0, self.k_neighbor):# loop duyệt k lần để chọn ra k lân cận 
                min_index = 0 # gán CHỈ SỐ của khoảng cách nhỏ nhất là 0
                for h in range (1, len (distance_temp)): #duyệt từng phần tử trong (1)
                    min_temp = distance_temp[min_index] #gán khoảng cách nhỏ nhất là phần tử có CHỈ SỐ LÀ MIN_INDEX
                    if (distance_temp[h] < min_temp):#so sánh khoảng cách tại CHỈ SỐ h so với thằng có chỉ số MIN_INDEX hiện tại
                        min_index = h #cập nhật MIN_INDEX 
                index_KNeighbor_temp.append(min_index) #sau khi duyệt khoảng cách trong (1) --> append min_index mới nhất vào mảng
                distance_temp.pop(min_index) #sau đó xóa khoảng cách nhỏ nhất đi, để thực hiện lần tìm tiếp theo
            self.index_KNeighbor.append(index_KNeighbor_temp) #mảng (k lân cận) của n (len(self.index_of_minority)) số quan sát minority


    def generateSmote(self):
        '''trong trường hợp tỷ lệ T nhập vào > 100'''
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

    def smote (self):
        self.handleData()
        self.calculateAmount()
        self.calculateDistance()
        self.selectKNeighbor()
        self.generateSmote()


if __name__ == '__main__':

    #xử lý sao cho phù hợp với đầu vào hàm SMOTE 

    Df_dataset = pd.read_csv("D:\PYTHON\processImbalancedData\data2\DLchuanhoa3.csv")

    #khởi tạo đối tượng thuộc lớp smote và sử dụng hàm 
    a = SMOTE(Df_dataset,  300, 3)
    a.smote()
    print ("Số lượng mẫu minority ban đầu là: ", len(a.index_of_minority))
    print ("Số lượng mẫu synthetic cần sinh ra là: ", len(a.synthetic))
    
    #in các mẫu synthetic dưới dạng dataframe để dễ quan sát
    df = pd.DataFrame(a.synthetic)
    print (df)




