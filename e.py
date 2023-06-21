#LM SETUP
import transformers
from torch import tensor
tokenizer = transformers.T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")

g = lambda sentence : tokenizer.batch_decode(model.generate(input_ids = tensor(tokenizer([sentence])['input_ids'])))

#PROMPT SETUP
'''
These are four of the six prompt formats used in the ScoNe paper, specifically the ones that seem to work for UnifiedQA AND aren't geared toward self-rationalization.
'''
prompts = [
	lambda premise, hypothesis : f'Is it true that if {premise.lower().strip(".")}, then {hypothesis.lower().strip(".")}?',
	lambda premise, hypothesis : f'Assume that {premise.lower().strip(".")}. Is it then definitely true that {hypothesis.lower().strip(".")}? Answer yes or no.',
	lambda premise, hypothesis : f'If {premise.lower().strip(".")}, then {hypothesis.lower().strip(".")}. Is this true?',
	lambda premise, hypothesis : f'P: {premise}\nQ:{hypothesis}\nYes, No, or Maybe?'
]

#DATASET SETUP
import pandas
dataset = dict()

partitions = ['no_negation', 'one_not_scoped', 'one_scoped', 'one_scoped_one_not_scoped', 'two_not_scoped', 'two_scoped']
for partition in partitions:
	dataset[partition] = pandas.read_csv(f'scone_nli/train/{partition}.csv')

def evaluate_partition(partition):
	accuracies = []

	for row in partition.iterrows():
		for prompt in prompts:
			p = prompt(row[1]['sentence1_edited'], row[1]['sentence2_edited'])
			expectation = (row[1]['gold_label_edited'] == 'entailment')

			response = g(p)[0]
			accuracies.append(("yes" in response.lower()) == expectation)

	print(f'{sum(accuracies)/len(accuracies)} ({"".join(str(int(a)) for a in accuracies)})')


for partition in partitions:
	print(f'{partition}\t', end='')
	evaluate_partition(dataset[partition])
