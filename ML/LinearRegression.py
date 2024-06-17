# 2024.06.12~

"""Regression is a statistical technique used to analyze and model the relationship between a dependent variable (also called the response or outcome variable) and one or more independent variables (also called predictors or explanatory variables)
y = mx + b
[1] https://data-marketing-bk.tistory.com/entry/Python-%EC%84%A0%ED%98%95%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D%EC%9D%84-%EC%9D%B4%EB%A1%A0-%EA%B2%B0%EA%B3%BC%ED%95%B4%EC%84%9D-%EA%B7%B8%EB%A6%AC%EA%B3%A0-%EC%BD%94%EB%93%9C%EA%B9%8C%EC%A7%80-Linear-Regression-Model
[2] https://medium.com/@rizwan-ai/simple-linear-regression-from-scratch-using-python-b4fa044c3793

"""

import numpy as np

class LinearRegression:
    def __init__(self):
        self.slope_ = None
        self.intercept_ = None
        self.residual_ = None
        self.RSS = None
        self.TSS = None
        self.r2score_ = None

    def fit(self, x, y):
        # Calculate the mean of the input (x) and output data (y)
        x_mean = np.mean(x)
        y_mean = np.mean(x)

        # Calculate the terms needed for the slope (b1) and intercept (b0) of the regression line
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        # Calculate the slope (b1) and intercept (b0) of the regression line (regression equation)
        self.slope_ = numerator / denominator
        self.intercept_ = y_mean - self.slope_ * x_mean

        y_pred = self.intercept_ + self.slope_ * x
        self.residual_ = y - y_pred

        self.RSS = np.sum(self.residual_ ** 2) # Residual sum of squares
        self.TSS = np.sum((y - y_mean) ** 2) # Total sum of squares
        self.r2score_ = 1 - (self.RSS / self.TSS)

    def predict(self, x):
        return self.intercept_ + self.slope_ * x