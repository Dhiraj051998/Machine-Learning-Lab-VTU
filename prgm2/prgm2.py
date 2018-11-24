"""
For a given set of training data examples stored in a .CSV file, implement and
demonstrate the Candidate-Elimination algorithm to output a description of the set
of all hypotheses consistent with the training examples.
"""
import csv


def get_domains(data):
    d = [set() for _ in data[0]]
    for x in data:
        for i, value in enumerate(x):
            d[i].add(value)
    return list(sorted(x) for x in d)


def consistent(hypothesis, sample):
    more_general_parts = []
    for x, y in zip(hypothesis, sample):
        mg = x == "?" or (x != "0" and (x == y or y == "0"))
        more_general_parts.append(mg)
    return all(more_general_parts)


def min_generalization(hypothesis, sample):
    new_hypothesis = list(hypothesis)
    for i, key in enumerate(hypothesis):
        if not consistent(hypothesis[i], sample[i]):
            new_hypothesis[i] = '?' if hypothesis[i] != '0' else sample[i]
    return [tuple(new_hypothesis)]


def min_specialization(hypothesis, sample, domain):
    hypothesis = list(hypothesis)
    results = []
    for i, key in enumerate(hypothesis):
        if hypothesis[i] == '?':
            for val in domain[i]:
                if sample[i] != val:
                    new_hypothesis = hypothesis.copy()
                    new_hypothesis[i] = val
                    results.append(tuple(new_hypothesis))
        elif hypothesis[i] == '0':
            new_hypothesis = hypothesis.copy()
            new_hypothesis[i] = '0'
            results.append(tuple(new_hypothesis))
    return results


dataset = []
with open('data.csv') as csvfile:
    lines = csv.reader(csvfile)
    for row in list(lines)[1:]:
        dataset.append(tuple(row))
print(dataset)
domains = get_domains(dataset)
domains = domains[:-1]
print(domains)
# Initialize G to the set of maximally general hypotheses in H
G = {("?",) * len(domains)}
# Initialize S to the set of maximally specific hypotheses in H
S = {("0",) * len(domains)}
k = 0
print("\n G[{0}]:".format(k), G)
print("\n S[{0}]:".format(k), S)
# For each training example d, do
for i in dataset:
    k += 1
    attributes, output = i[:-1], i[-1]
    # If d is a positive example
    if output == 'yes':
        # Remove from G any hypotheses inconsistent with d
        G = {g for g in G if consistent(g, attributes)}
        # For each hypothesis s in S that is not consistent with d
        for s in list(S):
            if not consistent(s, attributes):
                # Remove s from S
                S.remove(s)
                # Add to S all minimal generalizations h of s such that
                s_plus = min_generalization(s, attributes)
                # h is consistent with d, and some member of G is more general than h
                S.update([h for h in s_plus if any([consistent(g, h) for g in G])])
    else:
        # Remove from S any hypotheses inconsistent with d
        S = {s for s in S if not consistent(s, attributes)}
        for g in list(G):
            if consistent(g, attributes):
                G.remove(g)
                g_minus = min_specialization(g, attributes, domains)
                G.update([h for h in g_minus if any([consistent(h, s)
                                                     for s in S])])

    print("\n G[{0}]:".format(k), G)
    print("\n S[{0}]:".format(k), S)

"""
Output

[
('sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'yes'),
('sunny', 'warm', 'high', 'strong', 'warm', 'same', 'yes'), 
('rainy', 'cold', 'high', 'strong', 'warm', 'change', 'no'),
('sunny', 'warm', 'high', 'strong', 'cool', 'change', 'yes')
]

[['rainy', 'sunny'], ['cold', 'warm'], ['high', 'normal'], ['strong'], ['cool', 'warm'], ['change', 'same']]

 G[0]: {('?', '?', '?', '?', '?', '?')}

 S[0]: {('0', '0', '0', '0', '0', '0')}

 G[1]: {('?', '?', '?', '?', '?', '?')}

 S[1]: {('sunny', 'warm', 'normal', 'strong', 'warm', 'same')}

 G[2]: {('?', '?', '?', '?', '?', '?')}

 S[2]: {('sunny', 'warm', '?', 'strong', 'warm', 'same')}

 G[3]: {('sunny', '?', '?', '?', '?', '?'), ('?', 'warm', '?', '?', '?', '?'), ('?', '?', '?', '?', '?', 'same')}

 S[3]: {('sunny', 'warm', '?', 'strong', 'warm', 'same')}

 G[4]: {('sunny', '?', '?', '?', '?', '?'), ('?', 'warm', '?', '?', '?', '?')}

 S[4]: {('sunny', 'warm', '?', 'strong', '?', '?')}

"""
