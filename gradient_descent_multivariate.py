import numpy as np

def gradient_descent_multiple(x, y):
    m_curr = np.zeros(x.shape[1])  # Initialize coefficients to zeros
    b_curr = 0
    iterations = 10000
    n = len(y)
    learning_rate = 0.01

    for i in range(iterations):
        y_predicted = np.dot(x, m_curr) + b_curr
        cost = (1/n) * sum((y - y_predicted)**2)
        md = -(2/n) * np.dot(x.T, (y - y_predicted))
        bd = -(2/n) * sum(y - y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("coefficients {}, intercept {}, cost {} iteration {}".format(m_curr, b_curr, cost, i))

# Example with two independent variables
x = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Example data with an additional column of ones for the y-intercept
y = np.array([5, 7, 9, 11, 13])

gradient_descent_multiple(x, y)
