import sys

#LM SETUP
import transformers
from torch import tensor


model_dir = sys.argv[1] if len(sys.argv) > 1 else "google/flan-t5-xl"
cache_dir = sys.argv[2] if len(sys.argv) > 2 else "~/scratch/dummy"

#get models
config=transformers.AutoConfig.from_pretrained(model_dir, cache_dir=cache_dir, revision='main', use_auth_token=None)
model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_dir, from_tf=False, config=config, cache_dir=cache_dir, revision='main', use_auth_token=None)
tokenizer = transformers.T5Tokenizer.from_pretrained(model_dir)
#create easy-to-call function to pass everything into the model and directly get the response
g = lambda sentence : tokenizer.batch_decode(model.generate(input_ids = tensor(tokenizer([sentence])['input_ids'])))[0]

#get the data
from sconeUtils import dataset, prompts
prompts = {'A':prompts['A']}
testset = dataset['test']
#setup the per-instance evaluation method
def evaluate(scope, format, qa_function):
	accuracies = []
	for row in testset[scope].iterrows():
		prompt = prompts[format](row[1]['sentence1_edited'], row[1]['sentence2_edited'])
		expectation = (row[1]['gold_label_edited'] == 'entailment')

		response = qa_function(prompt)
		accuracies.append(("yes" in response.lower()) == expectation)

		print(f'{prompt} ({expectation}): {response}', file=sys.stderr)

	return accuracies

#setup dataframes
from pandas import DataFrame
results = DataFrame(None, testset, prompts)
results_agg = DataFrame(None, testset, prompts)

#loop over and evaluate the results for accuracy
for format in prompts:
	for scope in testset:
		results_raw = evaluate(scope, format, g)
		results[format][scope] = results_raw
		results_agg[format][scope] = sum(results_raw) / len(results_raw)
results.to_csv('results.csv')
results_agg.to_csv('results_agg.csv')
#loop over and evaluate the results for consistency
consistency_string = 'consistency'
for format in prompts:
	consistency_by_bundle = [all(z) for z in zip(*[row[1][format] for row in results.iterrows()])]
	consistency = sum(consistency_by_bundle)/len(consistency_by_bundle)
	consistency_string += f',{consistency}'
with open('results_agg.csv', 'a') as consistency_writer:
	consistency_writer.write(consistency_string)
