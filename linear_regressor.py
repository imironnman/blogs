# Loading required add-in packages
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

import math

def _plot_hist(data, ax2) :

    ax2.clear()
    sns.histplot(data['Error'], ax=ax2, bins=10)
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Histogram of Residuals')

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

def _get_inital_linepoint(data, xcol, ycol, type='simple') :
    '''
    Returns possible initial data point of regressor line
    
    Inputs : 
        data: (pandas.DataFrame) -> Data that needs to be fitted with regressor
        xcol: (list) -> IDV Column names
        ycol: (list) -> DV Column names
        type: (string) -> Type of Regressosr (simple/complex)
        
    Returns :
        x1_coordinates: (list) -> X1 Co-ordinates list
        x2_coordinates: (list) -> X2 Co-ordinates list
    '''

    x1_coordinates = []
    x2_coordinates = []
    for col in xcol :
        
        if type == "simple" :
            x1 = data[(data[col] < data[col].mean())][col].mean()
            y1 = data[(data[col] < data[col].mean())][ycol].mean().values[0]

            x2 = data[(data[col] >= data[col].mean())][col].mean()
            y2 = data[(data[col] >= data[col].mean())][ycol].mean().values[0]

        elif type == "complex" :
            pass

        x1_coordinates.append(x1)
        x2_coordinates.append(x2)

    x1_coordinates.append(y1)
    x2_coordinates.append(y2)

    return x1_coordinates, x2_coordinates

def _get_y_from_slope(x, slope, constant) :
    '''
    Returns Y value from X, Slope and Constant
    
    Inputs : 
        x: (float) -> Input X-value
        slope: (float) -> Constant value of slope
        constant: (float) -> Constant value of straight line

    Returns :
        y_value: (float)
    '''

    return slope * x + constant

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

def _optimize_coefficients(error_data, learning_rate, slope, constant, error_type='squared_error') :
    '''
    Returns Optimized Coefficients
    
    Inputs : 
        error_data: (pandas.DataFrame) -> Input dataframe error info
        learning_rate: (float) -> Slope of the line
        slope: (float) -> Slope from the regressor
        constant: (float) -> Constant from the regressor
        error_type: (str) -> Type of error metric. Default is 'squared_error'. 

    Returns :
        slope: (float) -> Updated Slope
        constant: (float) -> Updated constant
    '''

    n = error_data.shape[0]
    
    if error_type == 'squared_error':
        delta_slope = (-2/n) * np.sum(error_data['Error'] * error_data['X'])
        delta_constant = (-2/n) * np.sum(error_data['Error'])
    elif error_type == 'absolute_error':
        delta_slope = (-1/n) * np.sum(np.sign(error_data['Error']) * error_data['X'])
        delta_constant = (-1/n) * np.sum(np.sign(error_data['Error']))
        
    slope = slope - learning_rate * delta_slope
    constant = constant - learning_rate * delta_constant
    
    return slope, constant


def _plot_graph(data, x_coordinates_line, y_coordinates_line, ax, slope, constant):
    '''
    Plots Graph based on data points and line
    Returns None
    
    Inputs : 
        data: (pandas.DataFrame) -> Input dataframe error info
        x_coordinates_line: (list) -> X-Coordinate of line
        y_coordinates_line: (list) -> Y-Coordinate of line
        ax (matplotlib.axes) -> Axis of the plot
        slope: (float) -> Slope from the regressor
        constant: (float) -> Constant from the regressor

    Returns :
    '''

    ax.clear()
    ax.scatter(data['X'], data['y'])

    ax.plot(x_coordinates_line, y_coordinates_line, color='red', linestyle='--', linewidth=2)
    ax.text(0.05, 0.95, f"Line Equation: y = {slope:.2f}x + {constant:.2f}", 
            transform=ax.transAxes, fontsize=12, verticalalignment='top')

    ax.set_title("Linear Regression Fit")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


def _create_residual_plot(data, ax2):
    '''
    Creates Residual Plots based on Actuals and predicted
    
    Inputs : 
        data: (DataFrame) -> Input dataframe
        ax (matplotlib.axes) -> Axis of the plot

    Returns :
        None
    '''
    # Calculate the residuals
    data['Residual'] = data['Error'].values
    mean_residual = data['Residual'].mean()

    # Clear the existing plot and create new
    ax2.clear()
    ax2.scatter(data['y_pred'], data['Residual'])
    ax2.axhline(y=0, color='r', linestyle='--')  # Add a horizontal line at y=0
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residual')
    ax2.set_title('Residual Plot')
    ax2.text(0.05, 0.95, f"Mean Residual = {mean_residual:.2f}", transform=ax2.transAxes, fontsize=12, verticalalignment='top')


def _calculate_errors(data) :

    mse = np.sum(np.square(data['Error']))/data.shape[0]
    rmse = math.sqrt(np.sum(np.square(data['Error'])))/data.shape[0]
    mae = np.sum(np.abs(data['y_true']-data['y_pred']))/data.shape[0]

    return mse, rmse, mae

def _update_plot(i):
    '''
    Update Plots based on latest co-ordinates
    
    Inputs : 
        i: (int) -> Iteration Count

    Returns :
        None
    '''

    global data, slope, constant, pearson_correlation
    error_data = _find_error(data=data, slope=slope, constant=constant, error_type='squared_error')
    slope, constant = _optimize_coefficients(error_data=error_data, learning_rate=learning_rate, slope=slope, constant=constant, error_type='squared_error')
    
    ax1.clear()
    ax1.scatter(data['X'], data['y'])
    
    x = np.linspace(data['X'].min(), data['X'].max(), num=10)
    y = slope * x + constant
    ax1.plot(x, y, color='red', alpha=0.5)
    
    r2 = _calculate_r2(y_true=error_data['y_true'], y_pred=error_data['y_pred'])
    adjusted_r2 = _calculate_adjusted_r2(r2=r2, n=data.shape[0], p=1) # Only 1 predictor (X) in this case

    mse, rmse, mae = _calculate_errors(data=error_data) # Only 1 predictor (X) in this case

    # _create_residual_plot(data=error_data, ax2=ax2)
    # _plot_hist(data=error_data, ax2=ax2)
    
    ax1.text(0.05, 0.95, f"Line Equation: y = {slope:.2f}x + {constant:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.90, f"MSE: {mse:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.85, f"RMSE: {rmse:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.80, f"MAE: {mae:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.75, f"R-squared: {r2:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.70, f"Adjusted R-squared: {adjusted_r2:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax1.text(0.05, 0.65, f"Pearson Correlation: {pearson_correlation:.2f}", transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    
    ax1.set_title("Linear Regression Fit")
    ax1.set_xlabel("X-axis")
    ax1.set_ylabel("Y-axis")


def _compute_pearson_correlation(data) :
    '''
    Computes Coefficient based on provided data
    
    Inputs : 
        data: (DataFarme) -> Input data

    Returns :
        r (float) -> Correlation Coefficient
    '''

    x_mean = data['X'].mean()
    y_mean = data['y'].mean()

    r =  np.sum((data['X'] - x_mean) * (data['y'] - y_mean)) / math.sqrt((np.sum((data['X'] - x_mean) ** 2)) * np.sum((data['y'] - y_mean) ** 2))

    return r





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
    data['y'] = np.random.random_sample(size=data_points) + 1
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
    data['y'] = np.where(data['X'] < 1.5, data['X'] * 1.5 + 0.5, np.square(data['X']) * 1.5 + 0.5)
elif relation == 7 :
    print('Creating Right skewed Linear data')
    data['y'] = data['X'] * 1.5 + 0.5
    data = data.sort_values('y')
    data['y'].iloc[-10:-1] = random.uniform(100, 200)
elif relation == 8 :
    print('Creating Left skewed Linear data')
    data['y'] = data['X'] * 1.5 + 0.5
    data = data.sort_values('y', ascending=False)
    data['y'].iloc[-10:-1] = random.uniform(0.01, 0.5)


pearson_correlation = _compute_pearson_correlation(data=data)
print('Correlation Coefficient is ', pearson_correlation)



slope = 0.5
constant = 1.5

print('Initializing with Slope of 0.5 and Constant of 1.5')

iteration_count = 1000
learning_rate = 0.2


# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 15))
fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.scatter(data['X'], data['y'])
ax1.set_title("Linear Regression Fit")
ax1.set_xlabel("X-axis")
ax1.set_ylabel("Y-axis")

line, = ax1.plot([], [], color='red', linestyle='--', linewidth=2)

ani = animation.FuncAnimation(fig, _update_plot, frames=iteration_count, interval=200)
ani.save('animation.gif', writer='imagemagick')

plt.tight_layout()
plt.show()