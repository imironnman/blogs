# Loading required add-in packages
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import math

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

def _do_acf(data, ax2) :
    plot_acf(data['Error'], ax=ax2)
    ax2.set_title("ACF of Residuals")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Residual")


def _do_pacf(data, ax2) :
    plot_pacf(data['Error'], ax=ax2)
    ax2.set_title("PACF of Residuals")
    ax2.set_xlabel("X-axis")
    ax2.set_ylabel("Residual")


def _calculate_r2(y_true, y_pred):
    '''
    Returns R2
    
    Inputs : 
        y_true: (pandas.DataFrame) -> Data that needs to be fitted with regressor
        y_pred: (list) -> IDV Column names
        
    Returns :
        r2 (float)
    '''

    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    return r2

def _calculate_adjusted_r2(r2, n, p):
    '''
    Returns Adjusted R2
    
    Inputs : 
        r2: (float) -> Fitted R2 Score
        n: (int) -> # of data points
        p: (int) -> # of features
        
    Returns :
        r2 (float)
    '''

    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return adjusted_r2

def _find_error(data, slope, constant, error_type='squared_error') :
    '''
    Returns Error between data points and line based on type of error metric specified
    
    Inputs : 
        data: (pandas.DataFrame) -> Input dataframe containing X and Y co-ordinates
        slope: (float) -> Slope of the line
        constant: (float) -> Constant from the regressor
        error_type: (str) -> Type of error metric. Default is 'squared_error'. 

    Returns :
        data: (pandas.DataFrame) -> DataFrame with error columns
    '''

    data['SimulatedY'] = slope * data['X'] + constant
    data['y_true'] = data['y']
    data['y_pred'] = data['SimulatedY']
    
    if error_type == 'squared_error':
        data['Error'] = (data['y_true'] - data['y_pred']).round(2)
    elif error_type == 'absolute_error':
        data['Error'] = np.abs(data['y_true'] - data['y_pred']).round(2)
    
    return data

def _plot_graph(i) :

    global data

    slope = -1.65
    constant = 2.5

    error_data = _find_error(data=data, slope=slope, constant=constant, error_type='squared_error')

    ax1.clear()
    ax1.scatter(data['X'], data['y'])
    
    x = np.linspace(data['X'].min(), data['X'].max(), num=10)
    y = slope * x + constant
    ax1.plot(x, y, color='red', alpha=0.5)
    
    r2 = _calculate_r2(y_true=error_data['y_true'], y_pred=error_data['y_pred'])
    adjusted_r2 = _calculate_adjusted_r2(r2=r2, n=data.shape[0], p=1) # Only 1 predictor (X) in this case

    # _create_residual_plot(data=error_data, ax2=ax2)
    _do_pacf(data=error_data, ax2=ax2)
    
    ax1.text(0.05, 0.95, f"Line Equation: y = {slope:.2f}x + {constant:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.90, f"R-squared: {r2:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.85, f"Adjusted R-squared: {adjusted_r2:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    
    ax1.set_title("Linear Regression Fit")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")

# Creating Dataframe with random values
data = pd.DataFrame()

print('Enter no of data points to generate : ')
data_points = int(input())

print('Enter Type of Data Generation : ')
print('1) Random')
print('2) Linear')
print('3) Cubic')
print('4) Sinusoidal')
print('5) Linear data with Outlier')
print('6) Linear data with Heteroscedasticity')
print('7) Right skewed Linear data')
print('8) Left skewed Linear data')

relation = int(input())


data['X'] = np.random.random_sample(size=data_points) + 1
if relation == 1 :
    print('Creating Random data between 0 and 1')
    data['y'] = np.random.random_sample(size=data_points)
elif relation == 2 :
    print('Creating Linear data as Y = 1.5*X + 0.5')
    data['y'] = data['X'] * 1.5 + 0.5
elif relation == 3 :
    print('Creating Cubic relation data as Y = 3*X^3 + 0*X^2 + 0.5*X + 5')
    data['y'] = (data['X'] ** 3) * 3 + (data['X'] ** 2) * 0 + data['X'] * 0.5 + 5
elif relation == 4 :
    print('Creating Cubic relation data as Y = 3*X^3 + 0*X^2 + 0.5*X + 5')
    data['y'] = np.sin(2 * np.pi * data['X'])
elif relation == 5 :
    print('Creating Linear data with Outlier')
    data['y'] = data['X'] * 1.5 + 0.5
    data = data.sort_values('y')
    data['y'].iloc[-2] = 75
    data['y'].iloc[-1] = 100
elif relation == 6 :
    print('Creating Linear data with Heteroscedasticity')
    data['y'] = np.where(data['X'] < 1.5, data['X'] * 1.5 + 0.5, data['X'] * 10 + 0.5)
elif relation == 7 :
    print('Creating Right skewed Linear data')
    data['y'] = data['X'] * 1.5 + 0.5
    data = data.sort_values('y')
    data['y'].iloc[-10:-1] = random.randint(100, 200)
elif relation == 8 :
    print('Creating Left skewed Linear data')
    data['y'] = data['X'] * 1.5 + 0.5
    data = data.sort_values('y', ascending=False)
    data['y'].iloc[-10:-1] = random.randint(0.01, 0.5)


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))  # Set up two subplots (one for fitting plot and one for residual plot)
# fig, ax1 = plt.subplots(figsize=(8, 6))  # Set up two subplots (one for fitting plot and one for residual plot)

ax1.scatter(data['X'], data['y'])
ax1.set_title("Linear Regression Fit")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")

line, = ax1.plot([], [], color='red', linestyle='--', linewidth=2)

ani = animation.FuncAnimation(fig, _plot_graph, frames=1, interval=200)
ani.save('animation.gif', writer='imagemagick')

plt.tight_layout()
plt.show()