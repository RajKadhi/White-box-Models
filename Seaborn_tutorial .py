#!/usr/bin/env python
# coding: utf-8

# ## Python Pandas - DataFrame

# A Data frame is a two-dimensional data structure, i.e., data is aligned in a tabular fashion in rows and columns.
# 
# ### Features of DataFrame
# 
# - Potentially columns are of different types
# - Size – Mutable
# - Labeled axes (rows and columns)
# - Can Perform Arithmetic operations on rows and columns

# Structure
# 
# Let us assume that we are creating a data frame with student’s data.
# 
# ![image.png](attachment:image.png)

# `pandas.DataFrame`
# 
# A pandas DataFrame can be created using the following constructor −
# 
# `pandas.DataFrame( data, index, columns)`
# 
# The parameters of the constructor are as follows −
# ![image.png](attachment:image.png)

# ## Create DataFrame
# 
# A pandas DataFrame can be created using various inputs
# 
# ### Create an Empty DataFrame
# 
# A basic DataFrame, which can be created is an Empty Dataframe.

# In[43]:


#import the pandas library and aliasing as pd
import pandas as pd
df = pd.DataFrame()
print(df)


# ## Create a DataFrame from Lists
# The DataFrame can be created using a single list or a list of lists.

# In[44]:


import pandas as pd
data = [1,2,3,4,5]
df = pd.DataFrame(data)
print(df)


# ## Visualization using Seaborn
#                                        
#                                   
# Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
# 
# Data visualization is one of the core skills in data science. In order to start building useful models, we need to understand the underlying dataset. You will never be an expert on the data you are working with, and will always need to explore the variables in great depth before you can move on to building a model or doing something else with the data. Effective data visualization is the most important tool in your arsenal for getting this done, and hence an critical skill for you to master.
# 
# In this tutorial series, we will cover building effective data visualizations in Python. We will cover the pandas and seaborn plotting tools in depth. We will also touch upon matplotlib

# #### We'll start by loading Pandas and the data:

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


fifa_filepath = r"C:\Users\ln2\Class_visualization\input\fifa.csv"
fifa_data = pd.read_csv(fifa_filepath, index_col='Date',parse_dates=True)


# ### Head() method displaying first few observations

# In[3]:


fifa_data.head(3)


# ### Indexing: Single Rows
# The simplest way to access a row is to pass the row number to the .iloc method. Note that first row is zero, just like list indexes.

# In[4]:


fifa_data.iloc[:,1]


# ### The other main approach is to pass a value from your dataframe's index to the .loc method:

# In[5]:


fifa_data.loc['1993-10-22']


# ### Lineplot using Seaborn
# 
# A line plot is a graph that shows frequency of data along a number line. It is best to use a line plot when comparing fewer than 25 numbers. It is a quick, simple way to organize data.

# In[6]:


# Set the width and height of the figure
plt.figure(figsize=(16,6))

# Line chart showing how FIFA rankings evolved over time 
sns.lineplot(data=fifa_data)


# ### Let's Look at line plots more deeper
# #### Step 1: Load the data into pandas datafraome 
# #### Step 2: Examine the data
# #### Step 3: Plot the data 
# 
# 
# ### Step 1: Loading the data

# In[7]:


# Path of the file to read
spotify_filepath = r"C:\Users\ln2\Class_visualization\input\spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)


# ### Step 2: Examining the data
# 
# We can print the first five rows of the dataset by using the head command that you learned about in the previous tutorial.

# In[8]:


# Print the first 5 rows of the data
spotify_data.head()


# Check now that the first five rows agree with the image of the dataset (from when we saw what it would look like in Excel) above.
# 
# Empty entries will appear as NaN, which is short for "Not a Number".
# 
# We can also take a look at the last five rows of the data by making only one small change (where .head() becomes .tail()):

# In[9]:


# Print the last five rows of the data
spotify_data.tail()


# Thankfully, everything looks about right, with millions of daily global streams for each song, and we can proceed to plotting the data!
# 
# ### Plot the data
# Now that the dataset is loaded into the notebook, we need only one line of code to make a line chart!

# In[10]:


# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)


# In[11]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of each song 
sns.lineplot(data=spotify_data)


# In[12]:


list(spotify_data.columns)


# ### Subsetting Data Plot

# In[13]:


# Set the width and height of the figure
plt.figure(figsize=(14,6))

# Add title
plt.title("Daily Global Streams of Popular Songs in 2017-2018")

# Line chart showing daily global streams of 'Shape of You'
sns.lineplot(data=spotify_data['Shape of You'], label="Shape of You")

# Line chart showing daily global streams of 'Despacito'
sns.lineplot(data=spotify_data['Despacito'], label="Despacito")

# Add label for horizontal axis
plt.xlabel("Date")


# ### BAR GRAPH
# 
# In this tutorial, we'll work with a dataset from the US Department of Transportation that tracks flight delays.
# 
# Each entry shows the average arrival delay (in minutes) for a different airline and month (all in year 2015). Negative entries denote flights that (on average) tended to arrive early. For instance, the average American Airlines flight (airline code: AA) in January arrived roughly 7 minutes late, and the average Alaska Airlines flight (airline code: AS) in April arrived roughly 3 minutes early.

# ### Step 1: Loading the data
# 
# As before, we load the dataset using the pd.read_csv command.

# In[14]:


# Path of the file to read
flight_filepath = r"C:\Users\ln2\Class_visualization\input\flight_delays.csv"

# Read the file into a variable flight_data
flight_data = pd.read_csv(flight_filepath, index_col="Month")


# ### Step 2: Examine the data
# Since the dataset is small, we can easily print all of its contents. This is done by writing a single line of code with just the name of the dataset.

# In[15]:


# Print the data
flight_data


# ### Bar chart
# 
# Say we'd like to create a bar chart showing the average arrival delay for Spirit Airlines (airline code: NK) flights, by month.

# In[16]:


# Set the width and height of the figure
plt.figure(figsize=(10,6))

# Add title
plt.title("Average Arrival Delay for Spirit Airlines Flights, by Month")

# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])

# Add label for vertical axis
plt.ylabel("Arrival delay (in minutes)")


# The commands for customizing the text (title and vertical axis label) and size of the figure are familiar from the previous tutorial. The code that creates the bar chart is new:

# In[17]:


# Bar chart showing average arrival delay for Spirit Airlines flights by month
sns.barplot(x=flight_data.index, y=flight_data['NK'])


# ### Heatmap
# We have one more plot type to learn about: heatmaps!
# 
# In the code cell below, we create a heatmap to quickly visualize patterns in flight_data. Each cell is color-coded according to its corresponding value.

# In[18]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

# Add title
plt.title("Average Arrival Delay for Each Airline, by Month")

# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=flight_data)

# Add label for horizontal axis
plt.xlabel("Airline")


# ### Correlation Heat map 
# 
# Where we first find te correlation of the whole data frame and then plot it as the heat map
# 

# In[19]:


# Set the width and height of the figure
plt.figure(figsize=(14,7))

corr = flight_data.corr()
ax = sns.heatmap(
    corr,vmin=-1,vmax=1,center=0,annot=True
)


# ### Load and examine the data
# 
# We'll work with a (synthetic) dataset of insurance charges, to see if we can understand why some customers pay more than others.

# In[20]:


# Path of the file to read
insurance_filepath = r"C:\Users\ln2\Class_visualization\input\insurance.csv"

# Read the file into a variable insurance_data
insurance_data = pd.read_csv(insurance_filepath)


# As always, we check that the dataset loaded properly by printing the first five rows.
# 

# In[21]:


insurance_data.head()


# In[22]:


insurance_data.rename({'sex':'gender'},axis =1,inplace=True)


# ### Scatter plots
# To create a simple scatter plot, we use the sns.scatterplot command and specify the values for:
# 
# `the horizontal x-axis (x=insurance_data['bmi']),` and
# `the vertical y-axis (y=insurance_data['charges']).`

# In[23]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# The scatterplot above suggests that body mass index (BMI) and insurance charges are positively correlated, where customers with higher BMI typically also tend to pay more in insurance costs. (This pattern makes sense, since high BMI is typically associated with higher risk of chronic disease.)
# 
# To double-check the strength of this relationship, you might like to add a regression line, or the line that best fits the data. We do this by changing the command to sns.regplot.
# 
# 

# In[24]:


sns.regplot(x=insurance_data['bmi'], y=insurance_data['charges'])


# ### Color-coded scatter plots
# We can use scatter plots to display the relationships between (not two, but...) three variables! One way of doing this is by color-coding the points.
# 
# For instance, to understand how smoking affects the relationship between BMI and insurance costs, we can color-code the points by 'smoker', and plot the other two columns ('bmi', 'charges') on the axes.
# 

# In[25]:


sns.scatterplot(x=insurance_data['bmi'], y=insurance_data['charges'], hue=insurance_data['smoker'])


# This scatter plot shows that while nonsmokers to tend to pay slightly more with increasing BMI, smokers pay MUCH more.
# 
# To further emphasize this fact, we can use the sns.lmplot command to add two regression lines, corresponding to smokers and nonsmokers. (You'll notice that the regression line for smokers has a much steeper slope, relative to the line for nonsmokers!)

# In[26]:


sns.lmplot(x="bmi", y="charges", hue="smoker", data=insurance_data)


# The sns.lmplot command above works slightly differently than the commands you have learned about so far:
# 
# Instead of setting `x=insurance_data['bmi']` to select the `'bmi'` column in insurance_data, we set x="bmi" to specify the name of the column only.
# Similarly, `y="charges"` and `hue="smoker"` also contain the names of columns.
# We specify the dataset with `data=insurance_data`.
# 
# 
# Finally, there's one more plot that you'll learn about, that might look slightly different from how you're used to seeing scatter plots. Usually, we use scatter plots to highlight the relationship between two continuous variables (like "bmi" and "charges"). However, we can adapt the design of the scatter plot to feature a categorical variable (like "smoker") on one of the main axes. We'll refer to this plot type as a categorical scatter plot, and we build it with the sns.swarmplot command.

# In[27]:


sns.swarmplot(x=insurance_data['smoker'],
              y=insurance_data['charges'])


# Among other things, this plot shows us that:
# 
# - on average, non-smokers are charged less than smokers, and
# - the customers who pay the most are smokers; whereas the customers who pay the least are non-smokers.

# ## Finding the overall whole distribution of data

# In[28]:


import seaborn as sns
g = sns.pairplot(insurance_data)
g.set(xticklabels=[])


# ## Distributions 
# We'll work with a dataset of 150 different flowers, or 50 each from three different species of iris (Iris setosa, Iris versicolor, and Iris virginica).
# ![image.png](attachment:image.png)

# ### Load and examine the data¶
# Each row in the dataset corresponds to a different flower. There are four measurements: the sepal length and width, along with the petal length and width. We also keep track of the corresponding species.

# In[29]:


# Path of the file to read
iris_filepath = r"C:\Users\ln2\Class_visualization\input\iris.csv"

# Read the file into a variable iris_data
iris_data = pd.read_csv(iris_filepath, index_col="Id")

# Print the first 5 rows of the data
iris_data.head()


# In[30]:


iris_data['Species'].value_counts()


# ### Histograms
# Say we would like to create a histogram to see how petal length varies in iris flowers. We can do this with the sns.distplot command.

# In[31]:


# Histogram 
sns.distplot(a=iris_data['Petal Length (cm)'], kde=False)


# We customize the behavior of the command with two additional pieces of information:
# 
# a= chooses the column we'd like to plot (in this case, we chose `'Petal Length (cm)'`).
# `kde=False` is something we'll always provide when creating a histogram, as leaving it out will create a slightly different plot.
# 
# ## Density plots
# The next type of plot is a kernel density estimate (KDE) plot. In case you're not familiar with KDE plots, you can think of it as a smoothed histogram.
# 
# To make a KDE plot, we use the sns.kdeplot command. Setting shade=True colors the area below the curve (and data= has identical functionality as when we made the histogram above).

# In[32]:


# KDE plot 
sns.kdeplot(data=iris_data['Petal Length (cm)'], shade=True)


# ### 2D KDE plots
# We're not restricted to a single column when creating a KDE plot. We can create a two-dimensional (2D) KDE plot with the sns.jointplot command.
# 
# In the plot below, the color-coding shows us how likely we are to see different combinations of sepal width and petal length, where darker parts of the figure are more likely.

# In[33]:


# 2D KDE plot
sns.jointplot(x=iris_data['Petal Length (cm)'], y=iris_data['Sepal Width (cm)'], kind="kde")


# Note that in addition to the 2D KDE plot in the center,
# 
# the curve at the top of the figure is a KDE plot for the data on the x-axis (in this case, `iris_data['Petal Length (cm)']`), and
# the curve on the right of the figure is a KDE plot for the data on the y-axis (in this case, `iris_data['Sepal Width (cm)']`).
# 
# 
# ### Color-coded plots
# For the next part of the tutorial, we'll create plots to understand differences between the species. To accomplish this, we begin by breaking the dataset into three separate files, with one for each species.
# 
# 

# In[34]:


# Paths of the files to read
iris_set_filepath = r"C:\Users\ln2\Class_visualization\input\iris_setosa.csv"
iris_ver_filepath = r"C:\Users\ln2\Class_visualization\input\iris_versicolor.csv"
iris_vir_filepath = r"C:\Users\ln2\Class_visualization\input\iris_virginica.csv"

# Read the files into variables 
iris_set_data = pd.read_csv(iris_set_filepath, index_col="Id")
iris_ver_data = pd.read_csv(iris_ver_filepath, index_col="Id")
iris_vir_data = pd.read_csv(iris_vir_filepath, index_col="Id")

# Print the first 5 rows of the Iris versicolor data
iris_ver_data.head()


# In the code cell below, we create a different histogram for each species by using the sns.distplot command (as above) three times. We use `label=` to set how each histogram will appear in the legend.

# In[35]:


# Histograms for each species
sns.distplot(a=iris_set_data['Petal Length (cm)'], label="Iris-setosa", kde=False)
sns.distplot(a=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", kde=False)
sns.distplot(a=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", kde=False)

# Add title
plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()


# In this case, the legend does not automatically appear on the plot. To force it to show (for any plot type), we can always use `plt.legend().`
# 
# We can also create a KDE plot for each species by using `sns.kdeplot` (as above). Again, `label=` is used to set the values in the legend.

# In[36]:


# KDE plots for each species
sns.kdeplot(data=iris_set_data['Petal Length (cm)'], label="Iris-setosa", shade=True)
sns.kdeplot(data=iris_ver_data['Petal Length (cm)'], label="Iris-versicolor", shade=True)
sns.kdeplot(data=iris_vir_data['Petal Length (cm)'], label="Iris-virginica", shade=True)

# Add title
plt.title("Distribution of Petal Lengths, by Species")


# **We learned how to create many different chart types. Now, let us organize your knowledge, before learning some quick commands that you can use to change the style of your charts.**
# 
# ### What have you learned?
# ![image.png](attachment:image.png)

# Since it's not always easy to decide how to best tell the story behind your data, we've broken the chart types into three broad categories to help with this.
# 
# - **Trends** - A trend is defined as a pattern of change.
#      - `sns.lineplot` - Line charts are best to show trends over a period of time, and multiple lines can be used to show trends in more than one group.
#      
# - **Relationship** - There are many different chart types that you can use to understand relationships between variables in your data.
#     - `sns.barplot` - Bar charts are useful for comparing quantities corresponding to different groups.
#     - `sns.heatmap` - Heatmaps can be used to find color-coded patterns in tables of numbers.
#     - `sns.scatterplot` - Scatter plots show the relationship between two continuous variables; if color-coded, we can also show the relationship with a third categorical variable.
#     - `sns.regplot` - Including a regression line in the scatter plot makes it easier to see any linear relationship between two variables.
#     - `sns.lmplot` - This command is useful for drawing multiple regression lines, if the scatter plot contains multiple, color-coded groups.
#     - `sns.swarmplot` - Categorical scatter plots show the relationship between a continuous variable and a categorical variable.
#     
# - **Distribution** - We visualize distributions to show the possible values that we can expect to see in a variable, along with how likely they are.
#     - `sns.distplot` - Histograms show the distribution of a single numerical variable.
#     - `sns.kdeplot` - KDE plots (or 2D KDE plots) show an estimated, smooth distribution of a single numerical variable (or two numerical variables).
#     - `sns.jointplot` - This command is useful for simultaneously displaying a 2D KDE plot with the corresponding KDE plots for each individual variable.
#     
# ### Changing styles with seaborn
# All of the commands have provided a nice, default style to each of the plots. However, you may find it useful to customize how your plots look, and thankfully, this can be accomplished by just adding one more line of code!
# 
# As always, we need to begin by setting up the coding environment. (This code is hidden, but you can un-hide it by clicking on the "Code" button immediately below this text, on the right.)
# 
# We'll work with the same code that we used to create a line chart in a previous tutorial. The code below loads the dataset and creates the chart.

# In[37]:


# Path of the file to read
spotify_filepath = r"C:\Users\ln2\Class_visualization\input\spotify.csv"

# Read the file into a variable spotify_data
spotify_data = pd.read_csv(spotify_filepath, index_col="Date", parse_dates=True)

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# We can quickly change the style of the figure to a different theme with only a single line of code.

# In[38]:


# Change the style of the figure to the "dark" theme
sns.set_style("dark")

# Line chart 
plt.figure(figsize=(12,6))
sns.lineplot(data=spotify_data)


# Seaborn has five different themes: (1)`"darkgrid"`, (2)`"whitegrid"`, (3)`"dark"`, (4)`"white"`, and (5)`"ticks"`, and you need only use a command similar to the one in the code cell above (with the chosen theme filled in) to change it.
# 
# The default theme is `"darkgrid"`.
# 
# In the upcoming exercise, you'll experiment with these themes to see which one you like most!

# ## Box plot using Seaborn
# 

# In[39]:


ax = sns.boxplot(x="smoker", y="charges", data=insurance_data)


# In[40]:


iris = sns.load_dataset("iris")
ax = sns.boxplot(data=iris, orient="h", palette="Set2")


# In[41]:


ax = sns.boxplot(x='smoker', y="charges", data=insurance_data)
ax = sns.swarmplot(x="smoker", y="charges", data=insurance_data, color=".25")


# In[ ]:





# In[ ]:




