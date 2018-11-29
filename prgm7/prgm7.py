"""
Write a program to construct a Bayesian network considering medical data. Use this
model to demonstrate the diagnosis of heart patients using standard Heart Disease
Data Set. You can use Java/Python ML library classes/API
"""
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel

from pgmpy.inference import VariableElimination  # Read the attributes

raw_data = pd.read_csv('dataset.csv')
print("Input Data and Shape")
print(raw_data.shape)
data = pd.DataFrame(raw_data, columns=['A', 'G', 'CP', 'BP', 'CH', 'ECG', 'HR', 'EIA', 'HD'])
data_train = data[: int(data.shape[0] * 0.80)]
variable_map = {
    "A": ["< 45", "45--55", "\geq 55"],
    "G": ["Female", "Male"],
    "CP": ["Typical", "Atypical", "Non-Anginal", "None"],
    "BP": ["Low", "High"],
    "CH": ["Low", "High"],
    "ECG": ["Normal", "Abnormal"],
    "HR": ["Low", "High"],
    "EIA": ["No", "Yes"],
    "HD": ["No", "Yes"]
}
# creating the baysian network
model = BayesianModel([('HD', 'CP'), ('G', 'BP'), ('A', 'CH'), ('G', 'CH'), ('HD', 'ECG'), ('HD',
                                                                                            'HR'), ('BP', 'HR'),
                       ('A', 'HR'), ('BP', 'HD'), ('CH', 'HD'), ('HD', 'EIA')])

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inferencing with Bayesian Network
print('\n Inferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)
    print(model.check_model())
# Computing the probability of HD given as a Variable.
print('\n1.Probability of HeartDisease given Gender=Female')

q = HeartDisease_infer.query(variables=['HD'], evidence={'G': 1})
print(q['HD'])

print('\n2. Probability of HeartDisease given BP= Low')

q = HeartDisease_infer.query(variables=['HD'], evidence={'BP': 1})
print(q['HD'])
