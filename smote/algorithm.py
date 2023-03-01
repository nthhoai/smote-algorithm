
import numpy as np
import pandas as pd
import random as rd
import math 

#PHƯƠNG PHÁP LẬP TRÌNH HƯỚNG ĐỐI TƯỢNG

class SMOTE:
    class_0 = [] #lưu chỉ số của ptu thuộc lớp minority
    number_0 = 0 #số lượng mẫu thuộc lớp minority
    number_sample = 0 #số lượng mẫu (thiểu số + đa số)
    num_att = 0 #số lượng thuộc tính
    distance_index = [] #lưu khoảng cách của từng điểm thiểu số tới tất cả điểm còn lại
    index_k = [] #lưu chỉ số và khoảng cách của k lân cận gần nhất
    path = 'D:\PYTHON\processImbalancedData\data2\DLchuanhoa2.csv' #đường dẫn file dữ liệu
    synthetic = pd.DataFrame(columns = ['school','sex', 'age','address'	,'famsize', 'Pstatus','Medu',
                        'Fedu','traveltime','studytime', 'failures'	,'schoolsup', 'famsup'	, 'paid','activities',
                'nursery', 'higher', 'internet','romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences',
        'G1', 'G2', 'Mjob_at_home', 'Mjob_health', 'Mjob_services', 'Mjob_teacher', 'Fjob_at_home', 'Fjob_health', 'Fjob_services', 'Fjob_teacher', 
        'R_course', 'R_home', 'R_reputation', 'G_father', 'G_mother', 'label']) #dataframe lưu các mẫu synthetic được tạo ra 


    def __init__(self):
        print ('Gọi hàm khởi tạo !')


    def handleFile(self):
        ''' XỬ LÝ FILE
        + Đầu vào: đường dẫn
        + Đầu ra: class_0 gồm chỉ số của các phần tử thuộc lớp minority + number_0 là số lượng mẫu thuộc minority + num_att là số thuộc tính
        '''
        self.df = pd.read_csv(self.path) 
        self.num_att = self.df.shape[1]
        for i in range (len(self.df)):
            if self.df['label'][i] == 0:
                self.class_0.append(i)
                self.number_0 = self.number_0 + 1
        self.number_sample = len(self.df)         


    
    def calculateAmount(self, T):
        ''' Tính số mẫu synthetic cần tạo ra
       + Đầu vào: number - là số lượng mẫu minority , t - là tỷ lệ % tăng mẫu so với ban đầu
       + Đầu ra: (t/100) * number
        '''
        flag = False
        if T < 100: #xét trường hợp tỷ lệ T nhập vào nhỏ hơn 100 --> chọn ngẫu nhiên trong lớp thiểu số tỷ lệ T tương ứng
            return True
        else: #ngược lại, sinh thêm lượng amount mẫu mới
            self.amount = int((T/100) * self.number_0)

    def Tless100(self):
        ''' HÀM NÀY DÙNG ĐỂ XỬ LÝ KHI TỶ LỆ T NHẬP VÀO < 100'''



    def calculateDistance(self):
        ''' TÍNH KHOẢNG CÁCH TỪ ĐIỂM GỐC ĐẾN TẤT CẢ CÁC ĐIỂM CÒN LẠI THUỘC LỚP MINORITY
        '''        
        for i in range (0, self.number_0): #tính khoảng cách từ 1 điểm thiểu số tới tất cả các điểm còn lại trong df
            distance = [] #lưu khoảng cách từ 1 điểm đc chọn đến n điểm còn lại
            self.point_or = self.class_0[i]
            for j in range (0, len(self.df)): #j chạy từ 0 đến tổng số điểm trong df
                square_sum = 0
                for h in range (0, self.num_att): #h duyệt từng thuộc tính (chiều) để tính tổng bình phương
                    square_sum = square_sum + math.pow ((self.df.iloc[self.point_or][h] - self.df.iloc[j][h]), 2)
                d = math.sqrt(square_sum) #căn bậc 2 tổng bình phương
                distance.append([j, d]) #mảng lưu khoảng cách từ điểm có chỉ số i đến điểm có chỉ số j
                #distance = tuple(distance)#vì d kiểu float, nên dùng mảng lưu d sau đó chuyển sang tuple
            self.distance_index.append(distance) 
            del distance


    def deleteSameElement(self):
        ''' XÓA PHẦN TỬ TRÙNG, VÌ NÓ CÓ TỪ ĐIỂM ĐẾN CHÍNH NÓ = 0 --> TỨC LÀ XÓA NHỮNG ĐIỂM NHƯ VẬY ĐI'''
        for i in range (len(self.distance_index)): #duyệt từng mẫu trong lớp thiểu số
            temp = self.distance_index[i]
            for j in range (0, len(temp)): #duyệt khoảng cách từ điểm thiểu số được chọn tới tất cả những điểm còn lại
                if temp [j][0] == i: 
                    temp.pop(j)
                    break #lệnh dừng khi đã tìm thấy ptu thỏa mãn điều kiện if


    def convertTuple(self):
        '''CHUYỂN HÓA CẤU TRÚC LƯU TRỮ TỪ MẢNG KHOẢNG CÁCH CỦA TỪNG ĐIỂM LỚP THIỂU SỐ [[vị trí phần tử cần tính, khoảng cách tính được]] SANG TUPLE ĐỐI VỚI'''
        for i in range (len(self.distance_index)):
            temp = self.distance_index[i]
            for j in range (0, self.number_sample-1): 
                temp [j] = tuple (temp[j])
    

            
    def selectPointK (self, k):
        '''chọn ra k lân cận gần nhất cho từng mẫu lớp thiểu s
        + đầu vào: k 
        + đầu ra: index_k lưa k chỉ số và khoảng cách của k lân cận gần nhất đối với lần lượt từng mẫu thiểu số theo thứ tự 
        trong mảng class_0
        '''
        for h in range (0, self.number_0): #duyệt từng mẫu trong lớp thiểu số
            temp = self.distance_index[h]
            temp_k = []
            for j in range (0, k):#duyệt k lần để chọn ra k lân cận
                index_temp = 0  
                #duyệt từng khoảng cách từ điểm đc chọn với những điểm còn lại
                for i in range (1, len(temp)): #len(temp) là vì qua mỗi vòng lặp, số phần tử - đi 1                      
                    min_temp = temp[index_temp][1]
                    if temp[i][1] < min_temp:
                        index_temp = i
                temp_k.append(temp[index_temp]) 
                #del as_list[index_temp]
                temp.pop(index_temp)
            self.index_k.append(temp_k)

    def generateSmote(self, k):
        '''trong trường hợp tỷ lệ T nhập vào > 100'''
        while self.amount != 0:
            for i in range (len(self.class_0)):
                point_or = rd.randint(0,(k-1))
                diff = self.df.iloc[self.class_0[i]] - self.df.iloc[self.index_k[i][point_or][0]] 
                new_point = self.df.iloc[self.class_0[i]] + diff * round(rd.uniform(0,1), 3)
                #self.synthetic.append(new_point)
                self.synthetic.loc[len(self.synthetic)] = new_point
                self.amount = self.amount - 1


    def smote(self, T, k):
        self.handleFile()
        self.calculateAmount(T)
        self.calculateDistance()
        self.deleteSameElement()
        self.convertTuple()
        self.selectPointK(k)
        self.generateSmote(k)



if __name__ == '__main__':
    a = SMOTE()
    a.smote(200, 3)
    print ("##############################################")
    print ("Lớp thiểu số ban đầu: ", len(a.class_0))
    print ("Lớp thiểu số sau khi sinh mẫu: ", len(a.synthetic))
    print ("IN MẪU SYNTHETIC: ")
    print (a.synthetic)






        















    
