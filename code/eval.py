from collections import OrderedDict
import argparse
from sklearn import metrics as sklearn_metrics
import json

from seqeval.metrics import accuracy_score, classification_report
from seqeval.scheme import IOB2

from squad_evaluate import *

def read_file(file):
    dict_result = OrderedDict()
    
    with open(file, 'r', encoding='utf-8') as fr:
        for i, line in enumerate(fr.readlines()):
            sp_line = line.replace('\n', '').split('\t')
            if i == 0:
                list_header = sp_line
                for key in sp_line:
                    dict_result[key] = []
            else:
                for key, element in zip(list_header, sp_line):
                    dict_result[key].append(element)
    
    return dict_result, len(dict_result[key])


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_fpath',
                        type=str, default='',, help='test file')
    parser.add_argument('--pred_fpath',
                        type=str, default='', help='prediction file')
    parser.add_argument('--output_fpath',
                        type=str, default='output.tsv', help='output file')
    parser.add_argument('--is_output_dict', action='store_true')
    parser.add_argument('--is_ner', action='store_true')
    parser.add_argument('--is_korquad', action='store_true')

    return parser.parse_args()


def main():
    args = parse_argument()
    
    if args.is_korquad:        
        expected_version = 'KorQuAD_v1.0'
        with open(args.test_fpath) as dataset_file:
            dataset_json = json.load(dataset_file)
            read_version = "_".join(dataset_json['version'].split("_")[:-1])
            if (read_version != expected_version):
                print('Evaluation expects ' + expected_version +
                      ', but got dataset with ' + read_version,
                      file=sys.stderr)
            dataset = dataset_json['data']

        predictions = {}
        with open(args.pred_fpath, 'r', encoding='utf-8') as prediction_file:
            for idx, line in enumerate(prediction_file):
                if idx == 0:
                    print(line)
                    continue
                json_obj = json.loads(line.strip())
                for k, v in json_obj.items():
                    predictions.update({k:v[0]})
            
        dict_eval_result = evaluate(dataset, predictions)

        if args.is_output_dict:
            with open(args.output_fpath, 'w', encoding='utf-8') as fw:
                json.dump(dict_eval_result, fw, ensure_ascii=False, indent=2)
        else:
            with open(args.output_fpath, 'w', encoding='utf-8') as fw:
                text_eval_result = json.dumps(dict_eval_result)
                fw.write(text_eval_result + '\n')
    elif args.is_ner:
        list_gold = []
        with open(args.test_fpath, 'r', encoding='utf-8') as fr:
            reads = fr.read().strip()
            for idx, line in enumerate(reads.split('\n')):
                if idx == 0: 
                    print(idx, line)
                    continue
                json_obj = json.loads(line.strip())
                for k, v in json_obj.items():
                    kv_obj = (k, v)
                    break
                list_gold.append( (kv_obj[0], kv_obj[1]) )

        list_pred = []
        with open(args.pred_fpath, 'r', encoding='utf-8') as fr:
            reads = fr.read().strip()
            for idx, line in enumerate(reads.split('\n')):
                if idx == 0: 
                    print(idx, line)
                    continue
                json_obj = json.loads(line.strip())
                for k, v in json_obj.items():
                    kv_obj = (k, v)
                    break
                list_pred.append( (kv_obj[0], kv_obj[1]) )
    
        list_gold_all, list_pred_all = [], []
        list_gold_all_2, list_pred_all_2 = [], []
        for (pred_id, preds), (gold_id, golds) in zip(list_pred, list_gold):
            list_gold_temp, list_pred_temp = [], []
            list_gold_temp_2, list_pred_temp_2 = [], []
            len_gold = len(golds)
#             for idx in range(len_gold):
#                 pred_e, gold_e = preds[idx], golds[idx]
            for pred_e, gold_e in zip(preds, golds):
                pred_token, pred_label = pred_e
                gold_token, gold_label = gold_e

                if len(pred_label) == 0:
                    list_pred_temp.append("O")
                else:
                    list_pred_temp.append(pred_label)
                    
                list_gold_temp.append(gold_label)
                if gold_label != 'O':
                    if len(pred_label) == 0:
                        list_pred_temp_2.append("O")
                    else:
                        list_pred_temp_2.append(pred_label)
                    list_gold_temp_2.append(gold_label)
                    continue

            list_pred_all.append(list_pred_temp)
            list_gold_all.append(list_gold_temp)
            
            list_pred_all_2.append(list_pred_temp_2)
            list_gold_all_2.append(list_gold_temp_2)
            
            
        with open(args.output_fpath, 'w', encoding='utf-8') as fw:
            # all-f1-score
            fw.write("="*20 + '  all-f1-score  ' + "="*20 + '\n')
            print('[*]', list_gold_all[:3])
            print('[**]', list_pred_all[:3])
            text_eval_result = classification_report(list_gold_all, list_pred_all, digits=4, suffix=False)            
            fw.write(text_eval_result + '\n\n\n')
            
            # character-level f1-score
            fw.write("="*20 + '  character-level f1-score  ' + "="*20 + '\n')
            text_eval_result_2 = classification_report(list_gold_all_2, list_pred_all_2, digits=4, mode='strict', scheme=IOB2)
            fw.write(text_eval_result_2 + '\n')
            
        # entity-level f1-score
#         a = classification_report(list_gold_all, list_pred_all, digits=4, mode='strict', scheme=IOB2)
    else:
    
#         dict_data_test, len_test = read_file(args.test_fpath)
#         dict_data_pred, len_pred = read_file(args.pred_fpath)

#         list_pred = []
#         list_gold = []
#         for idx in range(len_test):
#             id_test, label_test = dict_data_test['ID'][idx], dict_data_test['OUT_SEQ'][idx]
#             id_pred, label_pred = dict_data_pred['ID'][idx], dict_data_pred['LABEL'][idx]
#             if id_test == id_pred:
#                 list_pred.append(label_pred)
#                 list_gold.append(label_test)

        
        list_gold_temp = []
        with open(args.test_fpath, 'r', encoding='utf-8') as fr:
            reads = fr.read().strip()
            for idx, line in enumerate(reads.split('\n')):
                if idx == 0:
                    print(idx, line)
                    continue
                json_obj = json.loads(line.strip())
                for k, v in json_obj.items():
                    list_gold_temp.append( (k, v[-1]) )

        list_pred_temp = []
        with open(args.pred_fpath, 'r', encoding='utf-8') as fr:
            reads = fr.read().strip()
            for idx, line in enumerate(reads.split('\n')):
                if idx == 0:
                    print(idx, line)
                    continue
                json_obj = json.loads(line.strip())
                for k, v in json_obj.items():
                    list_pred_temp.append( (k, v[-1]) )
        
        list_gold, list_pred = [], []
        for (pred_id, pred_e), (gold_id, gold_e) in zip(list_pred_temp, list_gold_temp):
            if pred_id == gold_id:
                list_gold.append(gold_e)                
                list_pred.append(pred_e)
            else:
                print('[ERROR]', pred_id, pred_e, gold_id, gold_e )
    
        if args.is_output_dict:
            dict_eval_result = sklearn_metrics.classification_report(list_gold, list_pred, digits=4, output_dict=True, zero_division=0)
            with open(args.output_fpath, 'w', encoding='utf-8') as fw:
                json.dump(dict_eval_result, fw, ensure_ascii=False, indent=2)
        else:
            text_eval_result = sklearn_metrics.classification_report(list_gold, list_pred, digits=4, output_dict=False, zero_division=0)
            with open(args.output_fpath, 'w', encoding='utf-8') as fw:
                fw.write(text_eval_result + '\n')
    
            

if __name__ == "__main__":
    main()