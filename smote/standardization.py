import pandas as pd
import numpy as np

df = pd.read_csv('d:\python\processimbalanceddata\data2\student-mat.csv', sep=";")
df["school"].replace(["GP", "MS"], [1,0], inplace = True)
df["sex"].replace(["M", "F"], [0,1], inplace = True)
df["address"].replace(["R", "U"], [0,1], inplace = True)
df["famsize"].replace(["LE3", "GT3"], [1,2], inplace = True)
df["Pstatus"].replace(["A", "T"], [0,1], inplace = True)
df["schoolsup"].replace(["no", "yes"], [0,1], inplace = True)
df["famsup"].replace(["no", "yes"], [0,1], inplace = True)
df["paid"].replace(["no", "yes"], [0,1], inplace = True)
df["activities"].replace(["no", "yes"], [0,1], inplace = True)
df["nursery"].replace(["no", "yes"], [0,1], inplace = True)
df["higher"].replace(["no", "yes"], [0,1], inplace = True)
df["internet"].replace(["no", "yes"], [0,1], inplace = True)
df["romantic"].replace(["no", "yes"], [0,1], inplace = True)



#chuyển đổi, mã hóa nhãn cột g3
df['label'] = np.where(df['G3'] > 10, 'yes', 'no')
df['label'].replace(['no', 'yes'], [0, 1], inplace = True)
#chuấn hóa biến số 10
dummies_Fjob = pd.get_dummies(df.Fjob)
#chuẩn hóa biến số 10
dummies_Fjob.rename(columns = {'at_home': 'Fjob_at_home', 'health':'Fjob_health', 'services':'Fjob_services',
                              'other' : 'Fjob_other', 'teacher':'Fjob_teacher'}, inplace=True )


#biến số 9
dummies_Mjob = pd.get_dummies(df.Mjob)
#đổi tên  9
dummies_Mjob.rename(columns = {'at_home': 'Mjob_at_home', 'health':'Mjob_health', 'services':'Mjob_services',
                              'other' : 'Mjob_other', 'teacher':'Mjob_teacher'}, inplace=True )


#biến số 11
dummies_reason = pd.get_dummies(df.reason)
#đổi tên  11
dummies_reason.rename(columns = {'home': 'R_home', 'reputation':'R_reputation', 'course':'R_course',
                              'other' : 'R_other'}, inplace=True)


# get the dummies and store it in a variable, biến người giám hộ guardian 
dummies_guardian = pd.get_dummies(df.guardian)
dummies_guardian.rename(columns = {'mother': 'G_mother', 'father':'G_father', 'other':'G_other',
                              }, inplace=True)

#print (dummies) 
# concatenate the dummies to original dataframe
merged = pd.concat([df, dummies_Mjob, dummies_Fjob, dummies_reason, dummies_guardian], axis='columns')


# drop the values
merged = merged.drop(['Mjob', 'Fjob', 'guardian', 'reason', 'Fjob_other', 'Mjob_other', 'R_other','G_other', 'G3'], axis='columns')
#merged.rename(columns = {'Pstatus': 'ptogether', 'mother':'gmother', 'father':'gfather'},
#               inplace = True )
merged.to_csv('d:\python\processimbalanceddata\data2\DLchuanhoa.csv')