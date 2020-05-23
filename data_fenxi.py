# -*- coding:utf-8 -*-
# names data set

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


from matplotlib.font_manager import FontProperties 

font = FontProperties(fname=r"./simsun/simsun.ttc", size=14) 


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

# 计时
start = time.time()
# 组合98年到08年的数据到一张表, 一个名为names的dataframe
years = range(1998, 2009)
pieces = []
columns = ['name', 'sex', 'births']
for year in years:
    path = './datasets/babynames/yob%d.txt' % year
    frame = pd.read_csv(path, names=columns)
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index=True)  # 组合10个小表，且重新编号
# print(names)
print(names.describe())
print(names.head())
# 组合表导出到csv文件
# names.to_csv(columns=['names','sex','births','year'], header=True, index=True,  path_or_buf='./data_set/1998to2008.csv')
print('↑--------------------part i --------------------')

# 按照sex对每年births进行求和显示
births_sex = names.pivot_table(values='births', index='sex', columns='year', aggfunc=sum)
births_year = names.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
print(births_sex)
print(births_year)  # 两者互为转置
print('↑--------------------part ii--------------------')

# groupby, 对dataframe进行分组
names = names.groupby(['year', 'sex'])  # groupby之后不是一个dataframe对象，而是一个特殊的groupby对象
print(names.births.sum())  # 该年该性别的出生数之和
print(names.births.size())  # 该年该性别的出生数类别数，即该年该性别的人名数

# 男女人口增长曲线
figure_data = names.births.sum().reset_index()  # 每年每个性别的出生人口数
print(figure_data.head())


plt.figure()
plt.title(u'男女人口增长曲线')
plt.plot(figure_data[figure_data.sex == 'M'].year, figure_data[figure_data.sex == 'M'].births, label='male')
plt.plot(figure_data[figure_data.sex == 'F'].year, figure_data[figure_data.sex == 'F'].births, label='female')
plt.scatter(figure_data[figure_data.sex == 'M'].year, figure_data[figure_data.sex == 'M'].births, label=None)
plt.scatter(figure_data[figure_data.sex == 'F'].year, figure_data[figure_data.sex == 'F'].births, label=None)
plt.legend()

# plt.subplot(1,2,1)
plt.show()




# 给dataframe增加一个prop（比例）列，其数值为每个姓名出生数在该年该性别中的比例
# 先定义计算比例的函数，然后应用（apply）到分组后的dataframe中
def add_prop(group):  # 送入数据为group对象
    births = group.births.astype(float)  # 不可以直接用float()对series进行转换
    group['prop'] = births / births.sum()
    return group


names = names.apply(add_prop)  # apply这个函数之后，names变回dataframe对象
print(names.head())
print('↑-------------------part iii--------------------')

# 取出names的每对sex/year组合的前1000个名字
group = names.groupby(['year', 'sex'])
top1000 = group.apply(lambda g: g.sort_values(by='births', ascending=0)[0:1000])
print(top1000)
# 把各小组的前1000个名字分为boys和girls两个子集
boys = top1000[top1000.sex == 'M']
girls = top1000[top1000.sex == 'F']
# 以year为每一行的名字，names为每一列的名字，数值取births构造数据透视表
total_births = top1000.pivot_table(values='births', index='year', columns='name', aggfunc=sum)
print(total_births)
# 从数据透视表中取出几个名字，并画其人数曲线
subset = total_births[['Abby', 'Emily', 'Taylor', 'Victoria']]
print(subset)
# 这里留有一个疑问，figure_data和subset均为dataframe对象，但是数据的调用方式完全不一样，如figure_data[year=='1999'].birth, subset['Emily'][1999]
# 自问自答，因为figure_data是最初类型的dataframe，即由很多series构成的dataframe，再加上一个从0开始的index；
# 而subset的数据透视表的dataframe（pivot_table），横坐标人名，纵坐标年名
plt.figure()
plt.plot(subset['Abby'])
plt.plot(subset['Emily'])
plt.plot(subset['Taylor'])
plt.plot(subset['Victoria'])
plt.title(u'年代/人名')

# plt.subplot(1,2,2)

plt.show()
print('↑-------------------part iv---------------------')

# 评估命名的多样性增长
# 1、计算前1000个名字所占的比例——做一张以年、性别为横纵轴，以前1000个名字的比例和为元素的数据透视表
table = top1000.pivot_table(values='prop', index='year', columns='sex', aggfunc=sum)
table.plot()
plt.title(u'评估命名的多样性增长')
plt.show()
# 2、计算前50%人口的名字数量，以2008年男孩和1998年男孩为例
boys2008 = boys[boys.year == 2008]
prop_cumsum = boys2008.sort_index(by='prop',
                                  ascending=0).prop.cumsum()  # 对boys2008这个dataframe以prop为依据进行降序排列，然后对prop这个series进行累计求和
print(prop_cumsum)
print(prop_cumsum.searchsorted(0.5))  # 找到插入0.5而不改变原顺序的index，因为index从0开始，故前50%人口的名字数量为打印出来的值+1，即109
boys1998 = boys[boys.year == 1998]
prop_cumsum = boys1998.sort_index(by='prop',
                                  ascending=0).prop.cumsum()  # 对boys1998这个dataframe以prop为依据进行降序排列，然后对prop这个series进行累计求和
print(prop_cumsum)
print(prop_cumsum.searchsorted(0.5))  # 找到插入0.5而不改变原顺序的index，因为index从0开始，故前50%人口的名字数量为打印出来的值+1，即70
# 将该方法应用到整个top1000这个dataframe上
diversity = top1000.groupby(['year', 'sex']).apply(
    lambda g: g.sort_index(by='prop', ascending=0).prop.cumsum().searchsorted(0.5) + 1)
print(diversity.head())
print(type(diversity))  # series, 双重index
diversity = diversity.unstack('sex')
print(diversity.head())
print(type(diversity))  # dataframe, 将原series的双重index中的sex这个index解压出来变成列名(columns), 变成一个dataframe
# 画前20%人口的名字数目增长曲线
plt.figure()
plt.plot(diversity['F'], label='Female')
plt.plot(diversity['M'], label='Male')
plt.legend()
plt.title(u'前20%人口的名字数目增长曲线')
plt.show()
print('↑-------------------part v----------------------')

# 名字中最后一个字母的revolution
last_letters = names.name.map(lambda n: n[-1])  # 取出names这个dataframe中的name这个series，然后对其中的所有元素应用‘取出最后一个字母’的函数映射
print(last_letters.head())
print(type(last_letters))
letter_table = names.pivot_table(values='births', index=last_letters, columns=['sex', 'year'], aggfunc=sum)
# print(letter_table.head())
letter_sum = letter_table.sum()
letter_prop = letter_table / letter_sum.astype(float)
print(letter_prop)
# 画柱状图，male和female分成两张图，以字母为横轴，不同的年份为同一点上的不同柱子
letter_prop_sub = letter_prop.reindex(columns=[1998, 2003, 2008], level='year')
fig, axes = plt.subplots(2, 1)  # 两个子图用不同的ax区分表示
plt.title(u'男性占比图')
letter_prop_sub['M'].plot(kind='bar', ax=axes[0], title='Male')
plt.title(u'女性占比图')
letter_prop_sub['F'].plot(kind='bar', ax=axes[1], title='Female')


print('↑-------------------part vi---------------------')

# 通用名字
all_names = top1000.name.unique()
mask = np.array(['lesl' in ele.lower() for ele in all_names]) #trut or false 组成的一个array， 若名字中含有lesl几个连续字母则为true
name_with_lesl = all_names[mask] # 含有lesl这几个连续字母的名字
filters = top1000[top1000.name.isin(name_with_lesl)] #用这几个名字过滤top1000这个dataframe中的其他名字数据
print(filters)
# 把这个过滤后的dataframe转成以性别和年为横纵轴的数据透视表
filters_table = filters.pivot_table(values='births', index='year', columns='sex', aggfunc=sum)
print(filters_table) #98年之后就只有女生起这些名字了
print('↑-------------------part vii--------------------')

plt.show()
# 显示运行时间
print('total time is %.2f s' % (time.time() - start))