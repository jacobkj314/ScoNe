#PROMPT SETUP
'''
These are four of the six prompt formats used in the ScoNe paper, specifically the ones that seem to work for UnifiedQA AND aren't geared toward self-rationalization.
'''
prompts = {
	'A' : (lambda premise, hypothesis : f'Is it true that if {premise.lower().strip(".")}, then {hypothesis.lower().strip(".")}?'),
	'B' : (lambda premise, hypothesis : f'Assume that {premise.lower().strip(".")}. Is it then definitely true that {hypothesis.lower().strip(".")}? Answer yes or no.'),
	'C' : (lambda premise, hypothesis : f'If {premise.lower().strip(".")}, then {hypothesis.lower().strip(".")}. Is this true?'),
	'D' : (lambda premise, hypothesis : f'P: {premise}\nQ:{hypothesis}\nYes, No, or Maybe?')
}

#SCOPE SETUP
from pandas import read_csv
dataset = {
        'train' : {scope:read_csv(f'scone_nli/train/{scope}.csv') for scope in ['no_negation', 'one_not_scoped', 'one_scoped', 'one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']},
	'test' : {scope:read_csv(f'scone_nli/test/{scope}.csv') for scope in ['no_negation', 'one_not_scoped', 'one_scoped', 'one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']}
}
