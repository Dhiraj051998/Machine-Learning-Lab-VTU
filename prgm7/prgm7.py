"""
Write a program to construct a Bayesian network considering medical data. Use this
model to demonstrate the diagnosis of heart patients using standard Heart Disease
Data Set. You can use Java/Python ML library classes/API
"""
import numpy as np
from urllib.request import urlopen
import urllib
import matplotlib.pyplot as plt
import sklearn as skl
import pandas as pd

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
