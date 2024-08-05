# %% [markdown]
# ## <center> <strong> Linear Regression </strong> </center>
# ## <center> <strong> Name: Haris Tahir Rana </strong> </center>

# %% [markdown]
# - The aim of this assignment is to familiarize you with Univariate Linear Regression and Multivariate Linear Regression.<br> <br>
# - I have used Python programming language to implement the code. <br> <br>
# - I did NOT use the scikit learn library to implement any function, unless mentioned otherwise.

# %% [markdown]
# ## Dataset Description
# 
# - The `temperatures.csv` dataset is designed for bias correction of next-day maximum and minimum air temperature forecasts produced by the Local Data Assimilation and Prediction System (LDAPS) operated by the Korea Meteorological Administration. <br> <br>
# - It covers summer seasons from 2013 to 2017 in Seoul, South Korea.
# 
# Dataset Summary:
# - **Feature Type:** Real
# - **Instances:** 7586
# - **Input Features:** 21 (including present-day temperature data, LDAPS model forecasts, and geographical information)
# - **Output:** Next-day maximum (Next_Tmax)
# 
# 
# We want to predict the next day temperature given the various features

# %% [markdown]
# #### Libraries used

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# #### Import Dataset

# %% [markdown]
# Here we load the `temperatures.csv` file using <b>Pandas</b> library

# %%
df = pd.read_csv('temperatures.csv')

# %% [markdown]
# ## Part 1 - Univariate Regression 
# 
# In this part, we will develop a univariate regression model using maximum temperature on present day `(Present_Tmax)` to predict the next day temperature `(Next_Tmax)`

# %% [markdown]
# #### 1) Feature Extraction:
# 
# We first extract the `Present_Tmax` column as input and the `Next_Tmax` column as output from the dataset.

# %%
x1 = df['Present_Tmax'].values.reshape(-1,1)
y1 = df['Next_Tmax'].values.reshape(-1,1)

# %% [markdown]
# #### 2) Splitting the dataset
# We then make a 70-30 split to divide the dataset into training and test datasets which will result in 4 variables: X_train, Y_train, X_test, Y_test

# %%
x1_train, x1_test, y1_train, y1_test = model_selection.train_test_split(x1, y1, test_size=0.3, random_state=42)

mean1 = np.mean(x1_train)
stdev1 = np.std(x1_train)

normalized_x1_train = (x1_train - mean1)/stdev1
normalized_x1_test = (x1_test - mean1)/stdev1

# %% [markdown]
# ### 3) Learn the parameters
# In this part, we will fit the linear regression parameters $\theta_0$ and $\theta_1$ to the given dataset.
# 
# The objective of linear regression is to minimize the cost function
# 
# $$ J(\theta) = \frac{1}{2m} \sum_{i=1}^m \left(\hat {y}^{(i)} - y^{(i)}\right)^2$$
# 
# where the hypothesis $\hat {y}^{(i)}$ is the predicted value for a given x and is given by the linear model and $m$ is the total number of datapoints
# $$ \hat {y} =  h_\theta(x) = \theta_0 + \theta_1 x$$
# 
# The parameters of our model are the $\theta_j$ values. These are
# the values we will adjust to minimize the cost $J(\theta)$. One way to do this is to
# use the batch gradient descent algorithm. In batch gradient descent, each
# iteration performs the update
# 
# $$ \theta_0 = \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m \left( \hat {y}^{(i)} - y^{(i)}\right)$$
# 
# $$ \theta_1 = \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m \left( \hat {y}^{(i)} - y^{(i)}\right)x^{(i)}$$
# 
# With each step of gradient descent, the parameters $\theta_0$ and $\theta_1$ come closer to the optimal values that will achieve the lowest cost $J(\theta)$.

# %%
def predict_SLR(normalized_array, theta0, theta1):  #This function will return the predicted value of y based on the given parameters
    y_predicted = []
    for value in normalized_array:    
        y_predicted = theta0 + (theta1 * normalized_array)
    return y_predicted

def cost_SLR(y_actual, y_hypothesis): 
    cost = 0
    for i in range (0,len(y_hypothesis)):
        cost = cost + ((y_hypothesis[i] - y_actual[i]) ** 2)
    print(cost)
    cost = (cost/(2 * len(y_hypothesis)))
    return cost

def sum_theta0 (y_actual, y_hypothesis):
    sum = 0
    for i in range(0, len(y_actual)):
        sum = sum + ((y_hypothesis[i] - y_actual[i]))
    return sum

def sum_theta1 (y_actual, y_hypothesis, normalized_array):
    sum = 0
    length = len(y_actual)
    for i in range(0, length):
        sum = sum + ((y_hypothesis[i] - y_actual[i]) * normalized_array[i])
    return sum
    
def gradient_descent_SLR(x,y,alpha,epochs):
    J = [] 
    theta0 = 0.5
    theta1 = 0.5
    for epoch in range(epochs):
        temp = predict_SLR(x, theta0, theta1)
        cost = cost_SLR(y, temp)
        J.append(cost)
        
        theta0 = theta0 - ((alpha/len(y))*(sum_theta0(y, temp)))
        theta1 = theta1 - ((alpha/len(y))*(sum_theta1(y, temp, x)))

    return theta0,theta1,J        

# %% [markdown]
# #### 4) Running Linear Regression
# 
# The hyperparameters (epochs and learning rate) are then tuned to get the best fit model for linear regression

# %%
n_epoch = 1000
alpha = 0.01

theta0, theta1, J = gradient_descent_SLR(normalized_x1_train, y1_train, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))

# %% [markdown]
# #### 5) Plotting results

# %% [markdown]
#  This should output a graph with the original dataset points and the linear regression model using the learned values of theta0 and theta1
# 

# %%
#DO NOT EDIT THIS CELL.
y1_pred_list_train = list()
for x in normalized_x1_train:
    y1_pred_list_train.append(predict_SLR(x, theta0, theta1))

plt.plot(x1_train, y1_train, 'bo', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(x1_train, y1_pred_list_train, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

y1_pred_list_test = list()
for x in normalized_x1_test:
    y1_pred_list_test.append(predict_SLR(x, theta0, theta1))

plt.plot(x1_test, y1_test, 'ro', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('Present_Tmax')
plt.plot(x1_test, y1_pred_list_test, '-')
plt.legend(['Test data', 'Linear regression'])
plt.show()

# %% [markdown]
# ### 6) Finding the Correlation
# Correlation is used to assess the association between features (input variables) and the target variable (output variable) in a dataset.
# 
# * A positive correlation indicates a direct, linear relationship: as one variable increases, the other tends to increase as well.
# * A negative correlation indicates an inverse, linear relationship: as one variable increases, the other tends to decrease.
# * A correlation of 0 suggests no linear relationship between the variables.
# 
# <br> 
# Here, we pick the top 5 features that have the best correlation with Next_Tmax. I have written them in the markdown block below the code block.

# %%
#the features parameter is all the 22 columns other than the output feature in the dataset
features = df.iloc[:,:21].values
y_train = df.iloc[:, 21].values

correlations = np.corrcoef(features, y_train, rowvar=False)[:21, -1]
column_names = df.columns[:21]
plt.figure(figsize=(18, 8))
plt.bar(range(1, 22), correlations, tick_label=column_names, color='skyblue')
plt.title('Correlation Coefficients between Features and Next_Tmax')
plt.xlabel('Features')
plt.ylabel('Correlation Coefficient')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# %% [markdown]
# #### Top 5 features:

# %% [markdown]
# 1) Present_Tmax
# 2) Present_Tmin
# 3) LDAPS_Tmax_lapse
# 4) LDAPS_Tmin_lapse
# 5) LDAPS_CC3

# %% [markdown]
# ### 7) Improving Performance
# 
# Now, we will try to improve performance of the model we have developed, i.e further reducing the cost, by selecting some other input feature instead of `Present_Tmax`, keeping the desired output as `Next_Tmax`
# 
# * We will also plot the final values of the cost function J for both models.

# %%
x2 = df['LDAPS_Tmax_lapse'].values.reshape(-1,1)
y2 = df['Next_Tmax'].values.reshape(-1,1)

x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(x2, y2, test_size=0.3, random_state=42)

mean2 = np.mean(x2_train)
stdev2 = np.std(x2_train)
normalized_x2_train = (x2_train - mean2)/stdev2
normalized_x2_test = (x2_test - mean2)/stdev2

n_epoch = 5000
alpha = 0.001
theta0, theta1, J = gradient_descent_SLR(normalized_x2_train, y2_train, alpha, n_epoch)
print('Predicted theta0 = %.4f, theta1 = %.4f, cost = %.4f' % (theta0, theta1, J[-1]))

# %%
y2_pred_list_train = list()
for x in normalized_x2_train:
    y2_pred_list_train.append(predict_SLR(x, theta0, theta1))

plt.plot(x2_train, y2_train, 'bo', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('LDAPS_Tmax_lapse')
plt.plot(x2_train, y2_pred_list_train, '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

y2_pred_list_test = list()
for x in normalized_x2_test:
    y2_pred_list_test.append(predict_SLR(x, theta0, theta1))

plt.plot(x2_test, y2_test, 'ro', ms=10, mec='k')
plt.ylabel('Next_Tmax')
plt.xlabel('LDAPS_Tmax_lapse')
plt.plot(x2_test, y2_pred_list_test, '-')
plt.legend(['Test data', 'Linear regression'])
plt.show()

# %% [markdown]
# ## Part 2: Multi-Variate Linear Regression 
# 
# We will now use a similar concept as in the previous part to train a multivariate regression model on the same dataset. Instead of using just one input feature, we will now use the `Top-5 input features` that we selected in the previous part. These features will be used to predict the next day temperature `(Next_Tmax)`

# %% [markdown]
# #### 1) Feature Extraction:
# 
# First, we extract the top five features from the dataset

# %%
X = df[['Present_Tmax', 'Present_Tmin','LDAPS_Tmax_lapse', 'LDAPS_Tmin_lapse', 'LDAPS_CC3' ]].values
Y = df['Next_Tmax'].values.reshape(-1,1)

# %% [markdown]
# #### 2) Splitting the dataset
# Then, we make a `70-30` split to divide the dataset into training and test datasets which will result in 4 variables: X_train, Y_train, X_test, Y_test

# %%
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)

scaler = StandardScaler()
Xtrain_normalized = scaler.fit_transform(X_train)
Xtrain_normalized_with_bias = np.insert(Xtrain_normalized, 0, 1, axis=1)
Xtest_normalized = scaler.fit_transform(X_test)
Xtest_normalized_with_bias = np.insert(Xtest_normalized, 0, 1, axis=1)

# %% [markdown]
# ## Learn the parameters for Multivariate Regression
# 
# In multivariate regression, we predict the output using multiple input features. The model has the form:
# 
# $$
# \hat{y} = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n
# $$
# 
# where:
# - $\hat{y}$ is the predicted value
# - $x_i$ represents each input feature
# - $\theta_i$ are the parameters of our model
# 
# The cost function for multivariate regression is an extension of the univariate case and is given by:
# 
# $$
# J(\Theta) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2
# $$
# 
# Here, $\Theta$ represents the parameter vector $(\theta_0, \theta_1, ..., \theta_n)$, and $m$ is the number of training examples.
# 
# To minimize the cost function $J(\Theta)$, we use a method such as gradient descent, where each parameter $\theta_j$ is updated as follows:
# 
# $$
# \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})x_j^{(i)}
# $$
# 
# - $\alpha$ is the learning rate
# - The summation is over all training examples
# 
# 
# With each iteration of gradient descent, the parameters $\Theta$ come closer to the optimal values that minimize the cost function $J(\Theta)$.

# %% [markdown]
# #### 3) Linear Regression:
# 
# We will implement <b>Gradient Descent</b> to train the model on the data. We will be using Mean Square Error as the cost function.

# %%
def predict_MLR(normalized_array, thetas): 
    Y_predicted = np.dot(normalized_array, thetas)
    return Y_predicted

def cost_MLR(Y_actual, Y_hypothesis):
    cost = 0
    for i in range (0,len(Y_hypothesis)):
        cost = cost + ((Y_hypothesis[i] - Y_actual[i]) ** 2)
    cost = (cost/(2 * len(Y_hypothesis)))
    print (cost) 
    return cost

def partial(Y_actual, Y_hypothesis, normalized_features_array):
    
    result = np.array(Y_hypothesis) - np.array(Y_actual)
    result = result * np.array(normalized_features_array)
    result = np.sum(result)
    return result

def gradient_descent_MLR(Xnormalized, Ytrain, alpha, epochs, size):
    J = []
    thetas = [0.5]*size
    thetas = np.array(thetas).reshape(-1,1)
    
    for i in range(epochs):
        temp = predict_MLR(Xnormalized, thetas)
        cost = cost_MLR(Ytrain, temp)
        J.append(cost)
        
        thetas = thetas - ((alpha/len(Ytrain))*(partial(Ytrain, temp, Xnormalized)))
        
    return thetas, J

# %% [markdown]
# #### 4) Run the Regression
# 
# For the specified number of epochs, we run the regression and store the costs and corresponding thetas for each epoch.

# %%
n_epoch = 1000
learning_rate = 0.001
thetas, J1= gradient_descent_MLR(Xtrain_normalized_with_bias, Y_train, learning_rate, n_epoch, 6) 

# %% [markdown]
# ### Visualizing the costs
# You can run the following cell to see how your costs change with each epoch.

# %%
plt.plot(np.arange(1,n_epoch),J1[1:])
plt.xlabel("number of epochs")
plt.ylabel("cost")

# %% [markdown]
# ### All input features
# 
# Now, we will use `ALL` the input features to predict `Next_Tmax`.
# 
# We extract the features and run the following code block again. Make note of the differences between this model and the one that only used `five features`.
# 

# %%
features = df.iloc[:,:21].values
output = df.iloc[:, 21].values.reshape(-1,1)

# %% [markdown]
# We perform 70-30 split on the dataset again.

# %%
features_train, features_test, output_train, output_test = model_selection.train_test_split(features, output, test_size=0.3, random_state=42)

scaler = StandardScaler()
normalized_features_train = scaler.fit_transform(features_train)
normalized_features_train_with_bias = np.insert(normalized_features_train, 0, 1, axis=1)
normalized_features_test = scaler.fit_transform(features_test)
normalized_features_test_with_bias = np.insert(normalized_features_test, 0, 1, axis=1)

# %% [markdown]
# We call the gradient descent function.

# %%
n_epoch = 1000
learning_rate = 0.001
thetas, J2= gradient_descent_MLR(normalized_features_train_with_bias, output_train, learning_rate, n_epoch, 22) 

# %% [markdown]
# Visualise the cost.

# %%
plt.plot(np.arange(1,n_epoch),J2[1:])
plt.xlabel("number of epochs")
plt.ylabel("cost")