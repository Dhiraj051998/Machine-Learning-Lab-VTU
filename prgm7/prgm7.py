"""
Write a program to construct a Bayesian network considering medical data. Use this
model to demonstrate the diagnosis of heart patients using standard Heart Disease
Data Set. You can use Java/Python ML library classes/API
"""
import pandas as pd

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination

dataset = pd.read_csv('dataset.csv')
print("Input Data and Shape")
print(dataset)
model = BayesianModel(
    [('HD', 'AGE'), ('HD', 'GENDER'), ('CP', 'AGE'), ('CHOLESTEROL', 'AGE'), ('HD', 'BP'), ('GENDER', 'CP')])

print(model.nodes())
print(model.edges())

"""

         CHOLESTEROL
              \
               \
                \
                 \
                  \
                   \        HD-----
                    \      / \     \
                     \    /   \     \
                      \  /     \     \
                      AGE   GENDER  BP
                       ^     /    
                       |    /
                       |   /                  
                       |  /
                       | /
                       CP
"""

model.fit(dataset, estimator=MaximumLikelihoodEstimator)

print('\n Inferencing with Bayesian Network:')

HeartDisease_infer = VariableElimination(model)
for cpd in model.get_cpds():
    print("CPD of {variable}:".format(variable=cpd.variable))
    print(cpd)
    print(model.check_model())
print('\n1.Probability of HeartDisease given Gender = Female')
q = HeartDisease_infer.query(variables=['HD'], evidence={'GENDER': 1})
print(q['HD'])

print('\n2. Probability of HeartDisease given BP = Low')
q = HeartDisease_infer.query(variables=['HD'], evidence={'BP': 1})
print(q['HD'])

print(model.get_independencies())

print(model.active_trail_nodes('HD'))

