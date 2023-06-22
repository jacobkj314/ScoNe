#LM SETUP
import transformers
from torch import tensor
tokenizer = transformers.T5Tokenizer.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")
#config=transformers.AutoConfig.from_pretrained('/uufs/chpc.utah.edu/common/home/u0403624/scratch/3b-working/0-0/out/unifiedqa-v2-t5-3b-1251000_negation_all_70_train_unifiedqa_test_unifiedqa', cache_dir='~/scratch/dummy', revision='main', use_auth_token=None)
#model = transformers.AutoModelForSeq2SeqLM.from_pretrained('/uufs/chpc.utah.edu/common/home/u0403624/scratch/3b-working/0-0/out/unifiedqa-v2-t5-3b-1251000_negation_all_70_train_unifiedqa_test_unifiedqa', from_tf=False, config=config, cache_dir='~/scratch/dummy', revision='main', use_auth_token=None)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained("allenai/unifiedqa-v2-t5-large-1251000")

g = lambda sentence : tokenizer.batch_decode(model.generate(input_ids = tensor(tokenizer([sentence])['input_ids'])))[0]


from utils import dataset, prompts
testset = dataset['test']

def evaluate(scope, format, qa_function):
	accuracies = []
	for row in testset[scope].iterrows():
		prompt = prompts[format](row[1]['sentence1_edited'], row[1]['sentence2_edited'])
		expectation = (row[1]['gold_label_edited'] == 'entailment')

		response = qa_function(prompt)
		accuracies.append(("yes" in response.lower()) == expectation)

	return accuracies


from pandas import DataFrame
results = DataFrame(None, testset, prompts)
results_agg = DataFrame(None, testset, prompts)

for format in prompts:
	for scope in testset['test']:
		results_raw = evaluate(scope, format)
		results[format][scope] = results_raw
		results_agg[format][scope] = sum(results_raw) / len(results_raw)


results.to_csv('results.csv')
results_agg.to_csv('results_agg.csv')
