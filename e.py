#SETUP

#LM SETUP
import transformers
from torch import tensor
tokenizer = transformers.T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
#config=transformers.AutoConfig.from_pretrained('/uufs/chpc.utah.edu/common/home/u0403624/scratch/3b-working/0-0/out/unifiedqa-v2-t5-3b-1251000_negation_all_70_train_unifiedqa_test_unifiedqa', cache_dir='~/scratch/dummy', revision='main', use_auth_token=None)
#model = transformers.AutoModelForSeq2SeqLM.from_pretrained('/uufs/chpc.utah.edu/common/home/u0403624/scratch/3b-working/0-0/out/unifiedqa-v2-t5-3b-1251000_negation_all_70_train_unifiedqa_test_unifiedqa', from_tf=False, config=config, cache_dir='~/scratch/dummy', revision='main', use_auth_token=None)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")

g = lambda sentence : tokenizer.batch_decode(model.generate(input_ids = tensor(tokenizer([sentence])['input_ids'])))

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
import pandas
dataset = dict()

scopes = ['no_negation', 'one_not_scoped', 'one_scoped', 'one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']
for scope in scopes:
	dataset[scope] = pandas.read_csv(f'scone_nli/test/{scope}.csv')

def evaluate(scope, prompt):
	accuracies = []
	for row in dataset[scope].iterrows():
		p = prompts[prompt](row[1]['sentence1_edited'], row[1]['sentence2_edited'])
		expectation = (row[1]['gold_label_edited'] == 'entailment')

		response = g(p)[0]
		accuracies.append(("yes" in response.lower()) == expectation)

	return accuracies

results = pandas.DataFrame(None, scopes, prompts)
results_agg = pandas.DataFrame(None, scopes, prompts)

for prompt in prompts:
	for scope in scopes:
		results_raw = evaluate(scope, prompt)
		results[prompt][scope] = results_raw
		results_agg[prompt][scope] = sum(results_raw) / len(results_raw)

results.to_csv('results.csv')
results_agg.to_csv('results_agg.csv')
