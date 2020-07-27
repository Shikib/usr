import bert_score
import sys

from nlgeval import compute_metrics

print(sys.argv[1])
fn = sys.argv[1]

gt = [e.replace("_go", "").replace("_eos", "").strip() for e in open("ap_data/valid_freq.tgt").readlines() ]
pred  = [e.replace("_go", "").replace("_eos", "").strip() for e in open(fn).readlines() ]

open("mt_tmp/pred.txt", "w+").writelines([e+"\n" for e in pred])
open("mt_tmp/gt.txt", "w+").writelines([e+"\n" for e in gt])
results = compute_metrics(hypothesis="mt_tmp/pred.txt", references=["mt_tmp/gt.txt"], no_skipthoughts=True, no_glove=True)
open("{0}.meteor".format(fn.split(".")[0]), "w+").write(str(results["METEOR"]))


pred = bert_score.score(pred, gt, device='gpu', model_type='roberta-base')
scores = pred[-1].tolist()
open("{0}.scores".format(fn.split(".")[0]), "w+").write(str(scores))
