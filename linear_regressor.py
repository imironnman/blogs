# Loading required add-in packages
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
        data['Error'] = data['y_true'] - data['y_pred']
    elif error_type == 'absolute_error':
        data['Error'] = np.abs(data['y_true'] - data['y_pred'])
    
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

def _update_plot(i):
    '''
    Update Plots based on latest co-ordinates
    
    Inputs : 
        i: (int) -> Iteration Count

    Returns :
    '''

    global data, slope, constant
    error_data = _find_error(data=data, slope=slope, constant=constant, error_type='squared_error')
    slope, constant = _optimize_coefficients(error_data=error_data, learning_rate=learning_rate, slope=slope, constant=constant, error_type='squared_error')
    
    ax.clear()
    ax.scatter(data['X'], data['y'])
    
    x = np.linspace(data['X'].min(), data['X'].max(), num=10)
    y = slope * x + constant
    ax.plot(x, y, color='red', alpha=0.5)
    
    r2 = _calculate_r2(error_data['y_true'], error_data['y_pred'])
    adjusted_r2 = _calculate_adjusted_r2(r2, data.shape[0], 1) # Only 1 predictor (X) in this case
    
    ax.text(0.05, 0.95, f"Line Equation: y = {slope:.2f}x + {constant:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.05, 0.90, f"R-squared: {r2:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    ax.text(0.05, 0.85, f"Adjusted R-squared: {adjusted_r2:.2f}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    
    ax.set_title("Linear Regression Fit")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")


# Creating Dataframe with random values

data = pd.DataFrame()
data['X'] = np.random.random_sample(size=40)
data['y'] = np.random.random_sample(size=40)

slope = 0
constant = 1

iteration_count = 1000
learning_rate = 0.1

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data['X'], data['y'])
ax.set_title("Linear Regression Fit")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")

line, = ax.plot([], [], color='red', linestyle='--', linewidth=2)

ani = animation.FuncAnimation(fig, _update_plot, frames=iteration_count, interval=200)
plt.show()