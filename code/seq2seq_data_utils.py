import json
import numpy as np
from typing import List, Dict, Callable, Tuple
from sklearn.metrics import f1_score
from transformers import EvalPrediction, PreTrainedTokenizer
from seq2seq.utils import (lmap,)


# # label2idx = {"정치": 0, "경제": 1, "사회": 2, "생활문화": 3, "세계": 4, "IT과학": 5, "스포츠": 6}
# ynat_label_list = ["정치", "경제", "사회", "생활문화", "세계", "IT과학", "스포츠"]

# def build_compute_metrics_fn_et5(tokenizer: PreTrainedTokenizer, _label_list=ynat_label_list) -> Callable[[EvalPrediction], Dict]:    
#     label_list = _label_list
#     print('[*]', label_list)

#     def non_pad_len(tokens: np.ndarray) -> int:
#         return np.count_nonzero(tokens != tokenizer.pad_token_id)

#     def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
#         pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
#         label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
#         pred_str = lmap(str.strip, pred_str)
#         label_str = lmap(str.strip, label_str)
#         return pred_str, label_str
# """
#     def ynat_metrics(pred: EvalPrediction) -> Dict:
#         pred_str, label_str = decode_pred(pred)
#         print(len(pred_str))
#         print(len(label_str))
#         print(pred_str[-1])
#         pred_str = [x if x in label_list else 'none' for x in pred_str]
#         label_str = [x if x in label_list else 'none' for x in label_str]
#         result = f1_score(y_true=label_str, y_pred=pred_str, average='macro')

#         return {
#             "F1(macro)": result,
#         }

#     compute_metrics_fn = ynat_metrics
#     return compute_metrics_fn
# """







def build_compute_metrics_fn_et5(tokenizer: PreTrainedTokenizer) -> Callable[[EvalPrediction], Dict]:  

    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)

        return pred_str, label_str
        

def read_json(filename):
    with open(filename) as fp:
        return json.load(fp)