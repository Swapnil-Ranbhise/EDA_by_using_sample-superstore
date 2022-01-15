#!/usr/bin/env python
# coding: utf-8

# # Exploratory data analysis.

# # Problem statement:
# 
# As a business manager try to find the weak ares where you can work to make more profit

# derive the remaining business problem by exploratory data analysis

# # Author:
# 
# Swapnil Ranbhise

# # Loading the dataset.

# In[1]:


# importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


my_df = pd.read_csv("SampleSuperstore.csv")
my_df.head()


# In[3]:


my_df.tail()


# In[4]:


my_df.info()


# In[5]:


my_df.isna().sum()


# In[6]:


my_df.describe()


# In[7]:


shape = my_df.shape
shape


# In[8]:


my_df.columns


# In[9]:


my_df.nunique()


# # Exploratory data analysis.

# In[10]:


fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10));

sns.countplot(my_df["Ship Mode"],ax=axs[0][0])
sns.countplot(my_df["Segment"],ax=axs[0][1])
sns.countplot(my_df["Region"],ax=axs[1][0])
sns.countplot(my_df["Category"],ax=axs[1][1])
axs[0][0].set_title("Ship Mode",fontsize=20)
axs[0][1].set_title("Segment",fontsize=20)
axs[1][0].set_title("Region",fontsize=20)
axs[1][1].set_title("Category",fontsize=20)

plt.tight_layout


# The segment, region, category and shipmodes these are the some of variable which is categorical in nature thats why the bar chart is used.

# In[11]:


plt.figure(figsize=(20,8))
sns.countplot(my_df["Sub-Category"], palette='Reds_r')
plt.title("Sub-Category",fontsize=20)


# the sub-category is also the factorial variable and to show the distribution of vatiable in graphical format i again used bar chart.
# here we can see the binder is the one sub-category which is having the highest values as compaire to oters.

# In[12]:


fig,axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10));

sns.distplot(my_df["Sales"],ax=axs[0][0])
sns.distplot(my_df["Quantity"],ax=axs[0][1])
sns.distplot(my_df["Discount"],ax=axs[1][0])
sns.distplot(my_df["Profit"],ax=axs[1][1])
axs[0][0].set_title("Sales",fontsize=20)
axs[0][1].set_title("Quantity",fontsize=20)
axs[1][0].set_title("Discount",fontsize=20)
axs[1][1].set_title("Profit",fontsize=20)

plt.tight_layout


# the variable which are mention above are all has data type as numerical, the profit is the only variable which is normally distributed and apart from that all the variables are rightly skiwed.

# In[13]:


plt.figure(figsize=(20,8))
sns.countplot(my_df['State'])
plt.title("Countplot of states",fontsize=20)
plt.xlabel('State',fontsize=20)
plt.ylabel('Count',fontsize=20)


# count of state wise distribution, showing which state has highest coint of records mention in data set.

# In[14]:


sns.pairplot(my_df)


# In[15]:


my_df.corr(method='pearson')


# In[16]:


plt.figure(figsize=(8,6))
sns.heatmap(my_df.corr(method='pearson'), cmap='rocket', annot=True)


# from the above to diagrame and correlation figure we can understand the which vatiable has highest correlation with each other
# and in the pearsn correlation values we can notice that there is only numerical data variable is calculated.
# from the above heatmap we can undertand that sales, quantity, discount and profit has high correlation thats why i make rest of EDA by taking these variable on priority basis.

# In[17]:



plt.figure(figsize=(8,6))
sns.lineplot('Discount','Profit', data=my_df, label='Discount')

plt.figure(figsize=(8,6))
sns.lineplot('Discount','Sales', data=my_df, label='Discount')

plt.figure(figsize=(8,6))
sns.lineplot('Quantity','Profit', data=my_df, label='Profit')

plt.figure(figsize=(8,6))
sns.lineplot('Quantity','Sales', data=my_df, label='Sales')

plt.show()


# there are 4 lineplots.
# profit-discount, sales-discount, profit-quantity, sales-quantity.
# first two diagram shows clear relation bitween sales, prfot, discount.
# we can see that when supplies gives high discount the sales increases but the profit goes down with higher discount.
# and rest of diagram shows that higher quantity = higher sales, but here there might be chances that higher sales of high quantity can be affected by the high discount.

# In[18]:


sales_profit_sum= my_df.groupby("Sub-Category")["Sales","Profit"].agg(['sum'])
sales_profit_sum.plot.bar(width=0.8, figsize=(14,6))
plt.title('Profit and Sales per sub-category', fontsize=20)
plt.xlabel('Sub-Category', fontsize=20)
plt.ylabel('Sales and Profit', fontsize=20)


# in the above diagram we can see that the most variable has high sales but low profit for instance chair and phones are those 2 subcategory who is having high sale but low profit.
# as it shown in earlier lineplot that higher sales can be cuased by high discount with low profit,
# the reason mention above can be one of factor which affect the sale and profit of most of the sub-category.

# In[19]:


plt.figure(figsize=(20,8))
sns.countplot(data=my_df, x='Sub-Category', hue='Region')
plt.title('Sales of Sub-Category in each region', fontsize=20)


# the sales of sub-category of each region is shown in the above diagram. the binders and paper is the two sub-category are having the highest sales figure as compare to other sub-category.
# where as the east and west region has the highest sales across all the region.

# In[20]:


my_df['Cost']= my_df['Sales']-my_df['Profit']
my_df['Profit %']= my_df['Profit']/my_df['Cost']*100
my_df.head()


# the above few diagram has shown the representaion of variable by taking sales and profit as independent variable but this explaination does not gives us the clear overview of profit.
# as we seen some variable has high sales but low profit and in this case discount can be the rason for low profit.
# thats why i calculate the profit ratio or percentage of profit.
# formula:- profit % = profit/cost*100
#           cost = sales-cost.
# 
# all the remaining diagram would be represented with profit % to have better understanding of analysis.

# In[21]:


plt.figure(figsize=(20,8))
sns.barplot(data=my_df, x='Segment', y='Profit %', hue='Sub-Category', palette='Paired')
plt.title('Profit % of Sub-Category in each segment', fontsize=20)
plt.grid()


# the earlier diagram were showing the porfit and sale differently and we were not able to compaire which area is weak and strong.
# thats why chair and phones wre having high sales but low profit.
# but from this diagram we can understand the profit percentage of each subcategory.
# here in the above diagram the labels, papers, fastners and envolopes are having the highest profit % over all across the segment.

# In[22]:


sns.barplot(data=my_df, x='Segment', y='Profit %')
plt.title('Profit for Segment')

plt.show()


# In[23]:


sns.barplot(data=my_df, x='Category', y='Profit %')
plt.title('Profit % for Category')


# In[24]:


sns.barplot(data=my_df, x='Region', y='Profit %')
plt.title('Profit % for each Region')


# In[25]:


sales_profit_sum= my_df.groupby("State")["Profit"].sum().nlargest(25)
sales_profit_sum.plot.bar(width=0.8, figsize=(14,6))
plt.title('Profit per State', fontsize=20)
plt.xlabel('State', fontsize=20)
plt.ylabel('Profit', fontsize=20)


# In[26]:


sales_profit_sum= my_df.groupby("State")["Sales"].sum().nlargest(25)
sales_profit_sum.plot.bar(width=0.8, figsize=(14,6))
plt.title('Sales per State', fontsize=20)
plt.xlabel('State', fontsize=20)
plt.ylabel('Sales', fontsize=20)


# # Conclussion:
# 
# 
# 1) standerd class, consumer, west region and office supplies category has a highest records within there data type.
# 2) the binders sub-category contains the highest records compare to other sub-category
# 

# 3) to present the numerical data type in graphical format the dencity plot has been used and as per the plot only the profit is normally distributed

# 4) for detailed analysis and to find a relation bitween multiple variable, the correlation methid is used with the help of the correlation we cuold see that the discount, sales, profit and quantity are those variable which has high correlation with each other.

# 5) by looking at the ineplot it is clearly evident that the high discount can increase sale till perticular point but with having low profit, the next 2 line plot shows that the high number of quantity can influance the sales and profit in positive manner.

# 6) the higher discount can be the cuase for high sales and low prift to understand and anlyze this situation in detailed manner the profit % parameter has been calculated, it shows the profit eprcentage of each variable.

# 7) earlier the binder and copier has highest sale among the sub-category but after plotting the profit % on  y axis we find that the label, paper, fastner and envolope is having the highest profit over all across the segment. with that as per profit % we have home office sgement has high profit % and the home supplies is one of the category which contains the mejority share of profit. apart from that the west region gain the maximum profit and the california is the state which sold the maximum product with large profit share.

# # Thank you.
