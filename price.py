import pandas as pd
def pay(usage_energy, types, month, fuel_fee_price=0):
    default = 0
    pay = 0
    environmental_fee_price = 7.3
    if types == '산업용(갑)저압':
        default = 5500
        if month == 6 or month == 7 or month == 8:
            pay = 80.9
        elif month == 11 or month == 12 or month == 1 or month == 2:
            pay = 79.2
        else:
            pay = 59.1

    elif types == '일반용(갑)저압':
        default = 6160
        if month == 6 or month == 7 or month == 8:
            pay = 105.6
        elif month == 11 or month == 12 or month == 1 or month == 2:
            pay = 65.1
        else:
            pay = 92.2

    elif types == '농사용(을)저압':
        default = 1150
        pay = 39.1

    elif types == '농사용(갑)':
        default = 360
        pay = 21.5

    elif types == '주택용전력':
        mid = 200
        late = 450
        if month == 7 or month == 8:
            mid = 300
            late = 450

        if usage_energy <= mid:
            default = 730
            pay = 78.2
        elif usage_energy > mid and usage_energy <= late:
            default = 1260
            pay = 147.2
        elif usage_energy > late and usage_energy <= 1000:
            default = 6060
            pay = 215.5
        else:
            default = 6060
            pay = 574.5

    elif types == '교육용(갑)저압':
        default = 5230
        if month == 6 or month == 7 or month == 8:
            pay = 96.8
        elif month == 11 or month == 12 or month == 1 or month == 2:
            pay = 84.0
        else:
            pay = 59.6

    elif types == '심야전력(갑)':
        if month == 11 or month == 12 or month == 1 or month == 2:
            pay = 76.7
        else:
            pay = 55.3

        default = 20 * pay

    elif types == '가로등(을)':
        default = 6290
        pay = 85.8

    return default + (environmental_fee_price * usage_energy) + (pay * usage_energy) + (fuel_fee_price * usage_energy)

#%%
types = list()
df_origin = pd.DataFrame()
# for month in range(1,13) :
month = 5
    #3, 7, 10, 12
if month == 4 or month == 6 or month == 9 or month == 11 :
    days = 30
elif month == 2 :
    days = 28
else :
    days = 31

# user_list = ['AA218호', 'AA1985호', 'AA314호', 'AA1822호', 'AA1334호', 'AA212호', 'AA492호'] # 각 계약별 대표되는 고객 #6월
# user_list = ['AA195호', 'AA116호', 'AA538호', 'AA14호', 'AA948호', 'AA263호', 'AA1122호'] # 각 계약별 대표되는 고객 #5
# user_list = ['AA195호', 'AA116호', 'AA172호', 'AA32호', 'AA739호', 'AA476호', 'AA783호'] #4
if month > 9 :
    path = f"/Users/choimunbong/Documents/경진대회__전력사용량1_나주시/nj_2018{month}.csv"
else :
    path = f"/Users/choimunbong/Documents/경진대회__전력사용량1_나주시/nj_20180{month}.csv"

user_list = list()
file = pd.read_csv(path, encoding="cp949")
df_origin2 = pd.DataFrame(file)
df_origin = pd.concat([df_origin, df_origin2])

for i in df_origin['계약종별'].unique().astype('str'):
    types.append(i)

for i in range(len(types)):
    print(f'type : {types[i]}')
    df1 = df_origin[((df_origin['시간'] % 100) % 15) == 0]
    df1 = df1[df1['계약종별'] == types[i]]  #계약 추출

    freq1 = df1['고객(가칭)'].value_counts().sort_values()
    print(f"mean : {freq1.median()}")
    mean_loss = freq1.mean() / (96 * days) # 전체 날짜 중 누락 수

    temp = len(df1['유효전력']) - df1['유효전력'].count() # 유효전력량에서의 누락
    temp_loss = temp / len(df1['유효전력'])

    total_loss = (temp_loss + (1.0 - mean_loss)) * 100

    #날짜가 많이 누락 될 경우 데이터 빈도의 평균 값보다 낮을 수 있다.
    print(f"데이터빈도 평균값 : {freq1.mean()}")
    print(f'total_loss : {total_loss} ')

    freq1 = df1['고객(가칭)'].unique()
    print(freq1[0])
    if types[i] == '농사용(갑)':
        user_list.append(freq1[0])
    else :
        user_list.append(freq1[0])
    print()
print(user_list)
print(types)
print(df_origin.columns)

temp_df = df_origin
print(temp_df.columns)
print(user_list)
# df = temp_df
for i in range(len(types)):
    # file = pd.read_csv(path, encoding="cp949")
    # file = pd.read_csv("/Users/choimunbong/Downloads/drive-download-20220330T064127Z-001/경진대회__전력사용량3_인천시(최종)/tb_201807.csv", encoding="cp949")
    # type = '산업용(갑)저압' # [산업용(갑)저압, 일반용(갑)저압, 농사용(을)저압, 주택용전력 ,교육용(갑)저압, 농사용(갑), 심야전력(갑)]
    # df = pd.DataFrame(file)
    # print(df.columns)
    # print(df)
    df = temp_df
    # try:
    df = df[((df['시간'] % 100) % 15) == 0]  #비정상 시간 데이터 제거
    df = df[df['계약종별'] == types[i]]  #계약 추출
    df1 = df[df['고객(가칭)'] == user_list[i]]  # 고객 추출
    ###############################
    '''
    날짜및 시간 보간법 넣을 것
    new_df = pd.concat([df.iloc[:2], new_row, df.iloc[2:]], ignore_index = True

    df.interpolate() --> 그라데이션 보간
    '''
    new_df = df1.drop(["시/도", "시/군/구", "고객(가칭)", '계약종별'], axis=1)
    df1 = df1.drop(["시/도", "시/군/구", "고객(가칭)", '계약종별'], axis=1)
    idx = 0

    # for i in range(0,len(df1)-1) :
    # #새로운 데이터 행에 삽입하는 방법
    #     the_day = df[i]['날짜']
    #     next_day = df[i+1]['날짜']
    #
    #     new_row = pd.DataFrame([['', '', '']], columns = df.columns)
    #     new_df = pd.concat([df.iloc[:2], new_row, df.iloc[2:]], ignore_index = True)
    # ################################

    # except:
    #     continue

    ######################################
    # print(df)
    usage = df1['유효전력'].sum()
    # print(usage)
    price = pay(usage_energy=usage, types=types[i], month=month)
    print(f"계약종별 : {types[i]}")
    print(f"price : {price}")
    # df = df[df['유효전력'] != None]
    # print(df)
    #가격 측정 및 데이터 빈도 측
    freq1 = df['고객(가칭)'].value_counts()  #.sort_values()
    freq = df['유효전력'].value_counts()  #.sort_values()
    # print(freq1)
    # print(f"type :{types[i]}")
    # print(f"데이터빈도 중간값 : {freq1.median()}")
    # print(f"데이터빈도 평균값 : {freq1.mean()}")

    temp = len(df['유효전력']) - df['유효전력'].count()
    mean_loss = freq1.mean() / (96 * days)
    temp_loss = temp / len(df['유효전력'])
    total_loss = temp_loss + (1.0 - mean_loss)

    print(f"평균 총합 결측률 : {total_loss * 100.0}%")
    #####################################################

    df1=df1.sample(frac=(1-total_loss))
    usage = df1['유효전력'].sum()
    price2 = pay(usage_energy=usage, types=types[i], month=month)

    # print(f"계약종별 : {types[i]}")
    print(f"loss_price : {price2}")
    print(f"loss_percent : {(1.0 - (price2/price))*100.0 }")
    print(f"loss_price : {(price2 - price)}")

    #
    # print(f"price : {price - (price * (total_loss / 100))}")

    # print(f"평균 시간데이터 결측률 : {(1.0-mean_loss)* 100}%")
    # print(f"유효전력데이터 결측률 : {(1.0 - temp_loss) * 100} %")

    # price = pay(default=default_arr[i], environmental_fee_price=env_fee_arr[i], usage_energy=)
    # df1['계약종별'] = i
    normalized_df = (df1 - df1.min()) / (df1.max() - df1.min())
    # print(normalized_df.head(5))
    print()
    # print(freq1.mode())
    # print(freq.mean())
    # print(df)
    # print(i)
