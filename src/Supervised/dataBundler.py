import json
from itertools import product
from random import shuffle

from sconeUtils import prompts, dataset

prompts = {k:v for k,v in prompts.items() if k=='A'} # # # # # ONLY USE ONE PROMPT (FOR NOW)

nli2qa = {'entailment':'YES', 'neutral':'NO'}

#train data
with open('../../data/unifiedqa_formatted_data/condaqa_train_unifiedqa.json', 'w') as writer:
        lines_to_write = []
        for format in prompts.values():
                devset = dataset['train']
                data = list(zip(*[devset[scope].iterrows() for scope in devset]))

                for row in range(len(data)):
                        bun_in = []; bun_ans = []
                        for scope in range(len(devset)):
                                bun_in.append(format(data[row][scope][1]['sentence1_edited'], data[row][scope][1]['sentence2_edited'])); bun_ans.append(nli2qa[data[row][scope][1]['gold_label_edited']])

                        #quick hacky bundling
                        bun_qa = list(zip(bun_in, bun_ans));shuffle(bun_qa)
                        ansset = {a for a in bun_ans}
                        lists = [[(q,a) for q,a in bun_qa if a==ans] for ans in ansset]

                        tupleBundles = list(product(*lists))

                        for tb in tupleBundles:
                                tb = list(tb); shuffle(tb)
                                tb_in, tb_ans = zip(*tb)

                                lines_to_write.append(
                                        json.dumps(
                                                {"input":list(tb_in), "answer":list(tb_ans)}
                                        )
                                )
        writer.write(
                '\n'.join(lines_to_write)
        )


#dev data
with open('../../data/unifiedqa_formatted_data/condaqa_dev_unifiedqa.json', 'w') as writer:
	lines_to_write = []
	for format in prompts.values():
		devset = dataset['dev']
		data = list(zip(*[devset[scope].iterrows() for scope in devset]))

		for row in range(len(data)):
			for scope in range(len(devset)):
				lines_to_write.append(
					json.dumps(
						{"input":format(data[row][scope][1]['sentence1_edited'], data[row][scope][1]['sentence2_edited']), "answer":nli2qa[data[row][scope][1]['gold_label_edited']]}
					)
				)
	writer.write(
		'\n'.join(lines_to_write)
	)

