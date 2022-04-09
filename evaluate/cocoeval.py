from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# read the results file
cocoEval = COCOEvalCap('../inference_result/magic_small_model_k_10_alpha_0.6_beta_2.0.json')
cocoEval.evaluate()
for metric, score in cocoEval.eval.items():
    print ('%s: %.3f'%(metric, score))
