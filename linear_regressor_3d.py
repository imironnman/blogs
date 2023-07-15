# Loading required add-in packages
import random
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


def _find_error(data, slope, constant) :
    '''
    Returns Error between data points and line based on type of error metric specified
    
    Inputs : 
        data: (pandas.DataFrame) -> Input dataframe containing X and Y co-ordinates
        slope: (float) -> Slope of the line
        constant: (float) -> Constant from the regressor

    Returns :
        data: (pandas.DataFrame) -> DataFrame with error columns
    '''

    data['SimulatedY'] = slope * data['X'] + constant
    data['Error'] = data['y'] - data['SimulatedY']
    
    return data

def _optimize_coefficients(error_data, learning_rate, slope, constant) :
    '''
    Returns Optimized Coefficients
    
    Inputs : 
        error_data: (pandas.DataFrame) -> Input dataframe error info
        learning_rate: (float) -> Slope of the line
        slope: (float) -> Slope from the regressor
        constant: (float) -> Constant from the regressor

    Returns :
        slope: (float) -> Updated Slope
        constant: (float) -> Updated constant
    '''

    n = error_data.shape[0]
    delta_slope = (-2/n) * np.sum(error_data['Error'] * error_data['X'])
    delta_constant = (-2/n) * np.sum(error_data['Error'])

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
    error_data = _find_error(data=data, slope=slope, constant=constant)
    slope, constant = _optimize_coefficients(error_data=error_data, learning_rate=learning_rate, slope=slope, constant=constant)
    
    ax.clear()
    ax.scatter(data['X'], data['y'], data['z'])
    
    x = np.linspace(data['X'].min(), data['X'].max(), num=10)
    y = slope * x + constant
    Z = np.full(x.shape, 0.5)
    ax.plot(x, y, Z, color='red', alpha=0.5)
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


# Creating Dataframe with random values
data = pd.DataFrame()
data['X'] = np.random.random_sample(size=40)
data['y'] = np.random.random_sample(size=40)
data['z'] = np.random.random_sample(size=40)

slope = 0
constant = 1

iteration_count = 1000
learning_rate = 0.1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ani = animation.FuncAnimation(fig, _update_plot, frames=iteration_count, interval=200)
plt.show()