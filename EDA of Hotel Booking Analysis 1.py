#!/usr/bin/env python
# coding: utf-8
# Have you ever wondered when the best time of year to book a hotel room is? Or the optimal length of stay in order to get the best daily rate? What if you wanted to predict whether or not a hotel was likely to receive a disproportionately high number of special requests? This hotel booking dataset can help you explore those questions!
This data set contains booking information for a city hotel and a resort hotel, and includes information such as when the booking was made, length of stay, the number of adults, children, and/or babies, and the number of available parking spaces, among other things. All personally identifying information has been removed from the data.
Explore and analyze the data to discover important factors that govern the bookings.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


df = pd.read_csv("downloads/Hotel Bookings.csv")


# In[109]:


df.head()


# In[110]:


df.tail()


# In[111]:


df.shape


# In[112]:


df.info()


# In[113]:


df.describe()


# In[11]:


# Let us create a copy of the data frame
df1 = df.copy()


# # Try to understand data

# In[115]:


df1.columns


# In[128]:


df1.nunique()


# In[129]:


df1['hotel'].unique()


# In[130]:


df1['is_canceled'].unique()


# In[131]:


df1['market_segment'].unique()


# In[132]:


df1['meal'].unique()


# In[133]:


df1['distribution_channel'].unique()


# In[134]:


df1['children'].unique()


# In[135]:


df1['arrival_date_year'].unique()


# # Data cleaning

# Finding the duplicate rows if any and remove those rows

# In[136]:


# no of rows of duplicate rows
df1[df1.duplicated()].shape


# In[137]:


# drop the duplicate values
df1.drop_duplicates(inplace=True)


# In[138]:


df1.shape


# # Missing values handling

# In[139]:


df1.isnull().sum().sort_values(ascending = False)


# In[140]:


# replacing the null values
df1[['company','agent']] = df1[['company','agent']].fillna(0)


#  This column 'children' has 0 as value which means 0 children were present in group of customers who made that transaction. So, 'nan' values are the missing values due to error of recording data.

# We will replace the null values under this column with mean value of children.
# 

# In[141]:


df1['children'].unique()


# In[142]:


df1['children'].fillna(df1['children'].mean(),inplace = True)


# Country column has datatype of string. We will replace the missing value with the mode of 'country' column.

# In[143]:


df1['country'].fillna('others', inplace = True)


# In[144]:


# let's see if all null values are removed or not
df1.isnull().sum().sort_values(ascending = False)


# Now let us remove the rows with value 0

# In[52]:


(df1[df1['adults']+df1['children']+df1['babies'] == 0]).shape


# In[145]:


df1.drop(df1[df1['adults']+df1['children']+df1['babies'] == 0].index, inplace = True)


# In adr column 5400 is very big outliers so we remove it

# In[146]:


df1 = df1[df1['adr']<5000]


# Let's convert datatype of some columns into appropriate form 

# In[148]:


df1[['children','agent','company']] = df1[['children','agent','company']].astype('int64')


# In[ ]:


df1['reservation_status_date'] = pd.to_datetime(df1['reservation_status_date'], format = '%Y-%m-%d')


#  Let us add some important columns/ data wrangling

# In[155]:


df1['total_stay'] = df1['stays_in_week_nights']+df1['stays_in_weekend_nights']


# In[158]:


df1['total_people'] = df1['adults']+df1['children']+df1['babies']


# # EDA
# 

# Lets first find the correlation between the numerical data.
# 
# Since, columns like 'is_cancelled', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month', 'is_repeated_guest', 'company', 'agent' are categorical data having numerical type. So we wont need to check them for correlation.
# 
# Also, we have added total_stay and total_people columns. So, we can remove adults, children, babies, stays_in_weekend_nights, stays_in_week_nights columns.

# In[162]:


temp_df1 = df1[[ 'lead_time','previous_cancellations','previous_bookings_not_canceled','booking_changes','days_in_waiting_list','adr','required_car_parking_spaces','total_of_special_requests','total_stay','total_people']]


# In[173]:


corrmat = temp_df1.corr()
f, ax = plt.subplots(figsize=(12,7))
sns.heatmap(corrmat,annot=True,fmt='.2f',annot_kws={'size':10},vmax=.8, square= True);


# 1) Total stay length and lead time have slight correlation which may means that for longer hotel stays people generally plan little before the the actual arrival.
# 
# 2) Adr is slightly correlated with total_people, which makes sense as more no. of people means more revenue, therefore more adr.

# Let's see the affect of length of stay on adr

# In[175]:


plt.figure(figsize=(12,6))
sns.scatterplot(y='adr',x='total_stay', data= df1)


# From the above plot we can see that as length of tottal_stay increases the adr decreases. This means for longer stay, the better deal for customer can be finalised.
# 
# 

# # Some Univariate Analysis

# Q1. Which agent makes most no.of bookings?

# In[227]:


da = pd.DataFrame(df1['agent'].value_counts()).reset_index().rename(columns={'count':'num_of_bookings'}).sort_values(by = 'num_of_bookings',ascending = False)
da.drop(da[da['agent'] == 0].index, inplace = True)   # 0 represents that booking is not made by an agent
da = da[:10]                                          # selecting top 10 agent's performance
plt.figure(figsize=(10,5))
plt.title('Agent_most_no_of_bookings')
sns.barplot(x='agent',y='num_of_bookings', data= da , order= da.sort_values('num_of_bookings',ascending = False).agent)
plt.show()


# Agent 9 has made the most no.of bookings.

# Q2. Which room type is the most demanding room type and which room type generates highest adr?

# In[254]:


fig,axes = plt.subplots(1,2,figsize=(18,8))

sns.countplot(ax = axes[0], x = df1['assigned_room_type'])
sns.boxplot(ax = axes[1], x = df1['assigned_room_type'], y = df1['adr'])
plt.show()



# As we can see that most demanded room type is A, but better adr rooms are of type H, G and C

# Q3. Which is the most preffered meal type of customers?

# In[243]:


plt.figure(figsize=(10,8)
plt.title('Most prefferd meal by customers')
sns.countplot(x=df1['meal'])
plt.show()


# As we can see above analysis most preffered meal by customers is BB(bed and breakfast)

# # Hotel wise analysis

# Q1.What is percentage of bookings in each hotel?

# In[287]:


grouped_by_hotel=df1.groupby('hotel')
d1=pd.DataFrame((grouped_by_hotel.size()/df1.shape[0])*100).reset_index().rename(columns= {0:'Booking %'})
plt.figure(figsize=(8,5))
sns.barplot(x=d1['hotel'],y=d1['Booking %'])
plt.show()


# Around  40% bookings are for Resort hotel and 60% bookings are for City hotel.
# 

# Q2. Which hotel seems to make more revenue ?

# In[303]:


d2 = grouped_by_hotel['adr'].agg(np.mean).reset_index().rename(columns={'adr':'avg_adr'}) # calculating avg adr(agg(np.mean))
plt.figure(figsize=(8,5))
sns.barplot(x=d2['hotel'], y=d2['avg_adr'])
plt.show()


# Avg adr of Resort hotel is slightly lower than that of City hotel. Hence, City hotel seems to be making slightly more revenue.

# Q3. Which hotel has higher lead time?

# In[305]:


d3=grouped_by_hotel['lead_time'].median().reset_index().rename(columns = {'lead_time':'median_lead_time'})
plt.figure(figsize=(8,5))
sns.barplot(x=d3['hotel'],y=d3['median_lead_time'])
plt.show()


# City hotel has slightly higher median lead time. Also median lead time is significantly higher in each case, this means customers generally plan their hotel visits way to early.

# Q4. What is preferred stay length in each hotel?

# In[309]:


not_canceled=df1[df1['is_canceled']==0]
d4=not_canceled[not_canceled['total_stay']<15]
plt.figure(figsize=(8,5))
sns.countplot(x=d4['total_stay'],hue=d4['hotel'])
plt.show()


# Most common stay length is less than 4 days and generally people prefer City hotel for short stay, but for long stays, Resort Hotel is preferred.

# Q5.Which hotel has longer waiting time?

# In[322]:


d5=pd.DataFrame(grouped_by_hotel['days_in_waiting_list'].agg(np.mean).reset_index().rename(columns={'days_in_waiting_list':'avg_waiting_period'}))
plt.figure(figsize=(8,5))
sns.barplot(x=d5['hotel'],y=d5['avg_waiting_period'])
plt.show()


# City hotel has significantly longer waiting time 

# Q6.Which hotel has higher booking cancellation rate?

# In[334]:


cancelled_data=df1[df1['is_canceled']==1]
cancel_grp = cancelled_data.groupby('hotel')
D1=pd.DataFrame(cancel_grp.size()).rename(columns={0:'total_cancelled_bookings'})


# Counting total number of bookings for each type of hotel
grouped_by_hotel=df1.groupby('hotel')
total_booking=grouped_by_hotel.size()
D2=pd.DataFrame(total_booking).rename(columns={0:'total_bookings'})
D3=pd.concat([D1,D2],axis=1)

#calculating cancel percentage
D3['cancel_%']= round((D3['total_cancelled_bookings']/D3['total_bookings'])*100,2)
D3


# In[337]:


plt.figure(figsize=(8,5))
sns.barplot(x=D3.index,y=D3['cancel_%'])
plt.show()


# Around 30% of city hotel of bookings got canceled.

# Q7.Which hotel has higher chance to their customer will return for another stay?

# In[340]:


# Selecting and counting repeated customers bookings
repeated_data=df1[df1['is_repeated_guest']==1]
repeat_grp=repeated_data.groupby('hotel')
D1=pd.DataFrame(repeat_grp.size()).rename(columns={0:'total_repeated_guests'})

# Counting total bookings
total_booking=grouped_by_hotel.size()
D2=pd.DataFrame(total_booking).rename(columns={0:'total_bookings'})
D3=pd.concat([D1,D2],axis=1)

#calculating repeat%
D3['repeat_%']=round((D3['total_repeated_guests']/D3['total_bookings'])*100,2)
D3


# In[346]:


plt.figure(figsize=(8,5))
sns.barplot(x=D3.index,y=D3['repeat_%'])
plt.show()


# Resort hotel has slightly higher repeat % than city hotel.

# # Distribution Channel wise Analysis

# Q1. Which is the most common channel for booking hotels?

# In[362]:


group_by_dc = df1.groupby('distribution_channel')
D5 = pd.DataFrame(round((group_by_dc.size()/df1.shape[0])*100,2)).reset_index().rename(columns={0:'Booking_%'})
plt.figure(figsize=(8,8))
data=D5['Booking_%']
labels=D5['distribution_channel']
plt.pie(x=data,autopct="%.2f%%",explode=[0.05]*5,labels=labels,pctdistance=0.5)
plt.title("Booking % by distribution channels", fontsize=14);                                                                                                 


# TA/TO is the most common channel for hotel booking

# Q2. Which channel is mostly used for early booking of hotels?

# In[367]:


group_by_dc = df1.groupby('distribution_channel')
D6 = pd.DataFrame(round(group_by_dc['lead_time'].median(),2)).reset_index().rename(columns={'lead_time':'median_lead_time'})
plt.figure(figsize=(7,5))
sns.barplot(x=D6['distribution_channel'],y=D6['median_lead_time'])
plt.show()


# TA/TO is mostly used for planning Hotel visits ahead of time
# 

# Q3. Which channel has longer average waiting time?

# In[368]:


D7 = pd.DataFrame(round((group_by_dc['days_in_waiting_list']).mean(),2)).reset_index().rename(columns={'days_in_waiting_list':'avg_waiting_time'})
plt.figure(figsize=(7,5))
sns.barplot(x= D7['distribution_channel'],y=D7['avg_waiting_time'])
plt.show()


# While booking via TA/TO one may have to wait a little longer to confirm booking of rooms.

# Q4. Which distribution channel brings better revenue generating deals for hotels?
# 

# In[369]:


group_by_dc_hotel = df1.groupby(['distribution_channel', 'hotel'])
D8=pd.DataFrame(round((group_by_dc_hotel['adr']).agg(np.mean),2)).reset_index().rename(columns={'adr':'avg_adr'})
plt.figure(figsize=(7,5))
sns.barplot(x= D8['distribution_channel'],y=D8['avg_adr'], hue= D8['hotel'])
plt.show()


# GDS channel brings higher revenue generating deals for City hotel, in contrast to that most bookings come via TA/TO.
# 
# Resort hotel has more revnue generating deals by direct and TA/TO channel. 
# 
# 

# # Analysis of booking cancellation

# Q1. Which significant distribution channel has highest cancellation percentage?

# In[372]:


A1=pd.DataFrame((group_by_dc['is_canceled'].sum()/group_by_dc.size())*100).drop(index='Undefined').rename(columns={0:'Cancel_%'})
plt.figure(figsize=(8,5))
sns.barplot(x=A1.index,y=A1['Cancel_%'])
plt.show()


#  Booking via TA/TO is 30% likely to get cancelled which is the highest

# Let us see what causes the cancelation of bookings of rooms by customers
# 
# One question can arise that may be longer waiting period or longer lead time causes the cancellation of bookings, let us check that.

# In[378]:


Waiting_bookings=df1[df1['days_in_waiting_list']!=0] #selecting bookings with non zero waiting time

fig,axes=plt.subplots(1,2, figsize=(18,8))
sns.kdeplot(ax=axes[0],x= 'days_in_waiting_list', hue='is_canceled', data=Waiting_bookings)
sns.kdeplot(ax=axes[1],x=df1['lead_time'], hue=df1['is_canceled'])
plt.show()


# Both the curve show that lead time or waiting period has no effect on cancellation of bookings

# Now let us check whether not getting allotted the same room type as demanded is the cause of cancellation fo bookings

# In[379]:


def check_room_allot(x):
  if x['reserved_room_type'] != x['assigned_room_type']:
    return 1
  else:
    return 0

df1['same_room_not_alloted'] = df1.apply(lambda x : check_room_allot(x), axis = 1)
grp_by_canc = df1.groupby('is_canceled')

D3 = pd.DataFrame((grp_by_canc['same_room_not_alloted'].sum()/grp_by_canc.size())*100).rename(columns = {0: 'same_room_not_alloted_%'})
plt.figure(figsize = (10,7))
sns.barplot(x = D3.index, y = D3['same_room_not_alloted_%'])
plt.show()


# The plot shows that not getting same room as demanded is not the case of cancellation of rooms.

# Lets see does not getting same room affects the adr.
# 

# In[381]:


plt.figure(figsize = (10,5))
sns.boxplot(x = 'same_room_not_alloted', y = 'adr', data = df1)
plt.show()


# So not getting same room do affects the adr, people who didn't got same room have paid a little lower adr, except for few exceptions.
# 

# # **Time wise analysis**

# In[417]:


d_month=df1['arrival_date_month'].value_counts().reset_index()
d_month.columns=['months','Number of guests']
d_month
months=['January','February','March','April','May','June','July','August','September','October','November','December']
d_month['months']=pd.Categorical(d_month['months'],categories=months, ordered=True)
d_month.sort_values('months').reset_index()


data_resort=df1[(df1['hotel'] == 'Resort Hotel') & (df1['is_canceled']== 0)]
data_city=df1[(df1['hotel'] == 'City Hotel') & (df1['is_canceled']== 0)]
resort_hotel= data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
final_hotel=resort_hotel.merge(city_hotel, on='arrival_date_month')
final_hotel.columns = ['month','price_for_resort','price_for_city_hotel']
final_hotel

resort_guest=data_resort['arrival_date_month'].value_counts().reset_index()
resort_guest.columns=['month','no of guests']
resort_guest

city_guest=data_city['arrival_date_month'].value_counts().reset_index()
city_guest.columns=['month','no of guests']
city_guest

final_guest=resort_guest.merge(city_guest, on='month')
final_guest.columns=['month','no of guests in resort','no of guests in city hotel']
final_guest
months=['January','February','March','April','May','June','July','August','September','October','November','December']
final_guest['month'] = pd.Categorical(final_guest['month'], categories=months, ordered=True)
final_guest=final_guest.sort_values('month').reset_index()

sns.lineplot(data=final_guest, x='month', y='no of guests in resort')
sns.lineplot(data=final_guest, x='month', y='no of guests in city hotel')
plt.legend(['Resort hotel','City hotel'])
plt.ylabel('Number of guest')
fig=plt.gcf()
fig.set_size_inches(15,10)



# the number of highest guest visits hotels in the august

# Now lets see which month results in high revenue

# In[418]:


resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
final_hotel = resort_hotel.merge(city_hotel, on = 'arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
final_hotel['month'] = pd.Categorical(final_hotel['month'], categories=months, ordered=True)
final_hotel = final_hotel.sort_values('month').reset_index()
sns.lineplot(data=final_hotel, x='month', y='price_for_resort')
sns.lineplot(data=final_hotel, x='month', y='price_for_city_hotel')
plt.legend(['Resort','City Hotel'])
plt.ylabel('adr')
fig = plt.gcf()
fig.set_size_inches(15,10)


# In[419]:


reindex = ['January', 'February','March','April','May','June','July','August','September','October','November','December']
df1['arrival_date_month'] = pd.Categorical(df1['arrival_date_month'],categories=reindex,ordered=True)
plt.figure(figsize = (15,8))
sns.boxplot(x = df1['arrival_date_month'],y = df1['adr'])
plt.show()


# Avg adr rises from beginning of year upto middle of year and reaches peak at August and then lowers to the end of year. But hotels do make some good deals with high adr at end of year also.

# Now let us check the trend of arrival_num and avg adr within a month.

# In[420]:


d6 = pd.DataFrame(not_canceled.groupby('arrival_date_day_of_month').size()).rename(columns = {0:'Arrival_num'})
d6['avg_adr'] = not_canceled.groupby('arrival_date_day_of_month')['adr'].agg(np.mean)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Plotting arrival num for each day of month
g = sns.lineplot(ax = axes[0],x = d6.index, y = d6['Arrival_num'])
g.grid()
g.set_xticks([1,7,14,21,28,31])
g.set_xticklabels([1,7,14,21,28,31])

# Plotting avg adr for each day of month
h = sns.lineplot(ax = axes[1],x = d6.index, y = d6['avg_adr'])
h.grid()
h.set_xticks([1,7,14,21,28,31])
h.set_xticklabels([1,7,14,21,28,31])

plt.show()


# We can see that graph Arrival_num has small peaks at regular interval of days. This can be due to increase in arrival weekend.
# 
# Also the avg adr tends to go up as month ends. Therefore charge more at the end of month.
# 
# 

# Let us divide our customers in three categories of single, couple and family/friends. then check their bookings.

# In[429]:


# Select single, couple, multiple adults and family
single = not_canceled[(not_canceled['adults']==1) & (not_canceled['children']== 0)&(not_canceled['babies']==0)]
couple = not_canceled[(not_canceled['adults']==2) & (not_canceled['children']== 0)&(not_canceled['babies']==0)]
family = not_canceled[not_canceled['adults']+not_canceled['children']+not_canceled['babies']>2]

reindex = ['January', 'February','March','April','May','June','July','August','September','October','November','December']

fig,ax = plt.subplots(figsize=(12,8))

for type in['single','couple','family']:
    d1=eval(type).groupby(['arrival_date_month']).size().reset_index().rename(columns={0:'arrival_num'})
    d1['arrival_date_month'] = pd.Categorical(d1['arrival_date_month'],categories=reindex, ordered=True)
    sns.lineplot(data= d1, x='arrival_date_month',y='arrival_num', label=type, ax=ax)
    
    plt.grid()
    plt.show
    


# Moslty bookings are done by couples(although we are not sure that they are couple as data doesn't tell about that)
# 
# 
# 

# # **Prediction of whether or not a hotel was likely to receive a disproportionately high number of special requests?**

# In[431]:


sns.boxplot(x='market_segment',y='total_of_special_requests', hue='market_segment',data=df1)
fig = plt.gcf()
fig.set_size_inches(15,10)


# All of market segment mostly have special request.

# In[435]:


#special request according to number of kid
df1['kids']=df1['children']+df1['babies']
sns.barplot(x='kids',y='total_of_special_requests',data=df1) # we use ci=10 parameter for line size 
fig=plt.gcf()
fig.set_size_inches(12,8)


# In[439]:


#special request according to number of kid
sns.barplot(x='adults',y='total_of_special_requests',data=df) # we use ci=10 parameter for line size 
fig=plt.gcf()
fig.set_size_inches(12,8)


# # **From which country the most guests are coming ?**
# 

# In[13]:


country_wise_guests=df1[df1['is_canceled']==0]['country'].value_counts().reset_index()
country_wise_guests.columns=['country','No of guests']
country_wise_guests


# In[3]:


pip install folium


# In[6]:


import folium
import plotly.express as px


# In[16]:


basemap = folium.Map()
guests_map = px.choropleth(country_wise_guests, locations = country_wise_guests['country'],color = country_wise_guests['No of guests'], hover_name = country_wise_guests['country'])
guests_map.show()


# In[17]:


grouped_by_country = df1.groupby('country')
d1=pd.DataFrame(grouped_by_country.size()).reset_index().rename(columns = {0:'Count'}).sort_values('Count', ascending = False)[:10]
sns.barplot(x=d1['country'],y=d1['Count'])
plt.show()


# Most guests are from Portugal and other Europian country

# # How long people stay at the hotels

# In[18]:


filter = df1['is_canceled'] == 0
data = df1[filter]
data.head()


# In[19]:


data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.head()


# In[20]:


stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[:, :3]
stay = stay.rename(columns={'is_canceled':'Number of stays'})
stay


# In[21]:


plt.figure(figsize = (10,5))
sns.barplot(x = 'total_nights', y = 'Number of stays',data= stay,hue='hotel')
plt.show()


# # **Most people prefer to stay at the hotels of <=5 days.**
