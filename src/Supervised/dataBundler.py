import json
from itertools import product
from random import shuffle

from sconeUtils import prompts, dataset

prompts = {k:v for k,v in prompts.items() if k=='A'} # # # # # ONLY USE ONE PROMPT (FOR NOW)

nli2qa = {'entailment':'YES', 'neutral':'NO'}

from sys import argv
enforce_scopes = '-enforce-scopes' in argv
even_scoped = ['no_negation', 'one_not_scoped', 'two_not_scoped', 'two_scoped']; odd_scoped = ['one_scoped', 'one_scoped_one_not_scoped']
num_bad_scope_pairs = 0
def scope_pair_is_ok(b_in, b_ans, scopes):
    if not ((sum(s in even_scoped for s in scopes) == 1) and (sum(s in odd_scoped for s in scopes) == 1)):
        print([b_in, b_ans, scopes])
        global num_bad_scope_pairs ; num_bad_scope_pairs += 1
        return False
    return True

#train data
with open('../../data/unifiedqa_formatted_data/condaqa_train_unifiedqa.json', 'w') as writer:
    lines_to_write = []
    for format in prompts.values():
        devset = dataset['train']
        data = list(zip(*[devset[scope].iterrows() for scope in devset]))

        for row in range(len(data)):
            bun_in = []; bun_ans = [] 
            bun_scopes = [] # # # # #
            for scope, scope_name in enumerate(devset):
                bun_in.append(format(data[row][scope][1]['sentence1_edited'], data[row][scope][1]['sentence2_edited'])); bun_ans.append(nli2qa[data[row][scope][1]['gold_label_edited']])
                bun_scopes.append(scope_name) # # # # #

            #quick hacky bundling
            bun_qa = list(zip(bun_in, bun_ans, bun_scopes));shuffle(bun_qa) # # # # # bun_qa = list(zip(bun_in, bun_ans));shuffle(bun_qa)
            ansset = {a for a in bun_ans}
            lists = [[(q,a,s) for q,a,s in bun_qa if a==ans] for ans in ansset] # # # # # lists = [[(q,a) for q,a in bun_qa if a==ans] for ans in ansset]


            tupleBundles = list(product(*lists))

            for tb in tupleBundles:
                tb = list(tb); shuffle(tb)
                tb_in, tb_ans, tb_scopes = zip(*tb) # # # # # tb_in, tb_ans = zip(*tb)

                if (not enforce_scopes) or scope_pair_is_ok(tb_in, tb_ans, tb_scopes): # # # # # This is the extra step to check that/whether 
                    lines_to_write.append(
                        json.dumps(
                            {"input":list(tb_in), "answer":list(tb_ans)}
                        )
                    )
    writer.write(
        '\n'.join(lines_to_write)
    )
if enforce_scopes:
    print(f'{num_bad_scope_pairs} pair{" was" if num_bad_scope_pairs==1 else "s were"} badly scoped')


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

