import sys

#LM SETUP
import transformers
from torch import tensor


model_dir = sys.argv[1] if len(sys.argv) > 1 else "allenai/unifiedqa-v2-t5-3b-1251000"
cache_dir = sys.argv[2] if len(sys.argv) > 2 else "~/scratch/dummy"

#tokenizer = transformers.T5Tokenizer.from_pretrained(model_dir)
config=transformers.AutoConfig.from_pretrained(model_dir, cache_dir=cache_dir, revision='main', use_auth_token=None)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_dir, from_tf=False, config=config, cache_dir=cache_dir, revision='main', use_auth_token=None)
tokenizer = transformers.T5Tokenizer.from_pretrained(model_dir)

g = lambda sentence : tokenizer.batch_decode(model.generate(input_ids = tensor(tokenizer([sentence])['input_ids'])))[0]


from sconeUtils import dataset, prompts
testset = dataset['test']

def evaluate(scope, format, qa_function):
	accuracies = []
	for i, row in enumerate(testset[scope].iterrows()):
		if i >=2:
			break
		prompt = prompts[format](row[1]['sentence1_edited'], row[1]['sentence2_edited'])
		expectation = (row[1]['gold_label_edited'] == 'entailment')

		response = qa_function(prompt)
		accuracies.append(("yes" in response.lower()) == expectation)

	return accuracies


from pandas import DataFrame
results = DataFrame(None, testset, prompts)
results_agg = DataFrame(None, testset, prompts)

for format in prompts:
	for scope in testset:
		results_raw = evaluate(scope, format, g)
		results[format][scope] = results_raw
		results_agg[format][scope] = sum(results_raw) / len(results_raw)


results.to_csv('results.csv')
results_agg.to_csv('results_agg.csv')
