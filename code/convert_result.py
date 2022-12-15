import argparse
import json
import re


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='')

    parser.add_argument('--pred_fpath',
                        type=str,
                        default='')
    
    parser.add_argument('--output_fpath',
                        type=str,
                        default='')
    
    return parser.parse_args()


def convertSeq(in_seqtext):
    list_result = []
    in_seqtext = ' ' + in_seqtext
    for e in re.finditer(' [^ ]+', in_seqtext):
        # print(e)
        result_text = e.group()
        # print('[*]', result_text)
        sp_result_text = result_text.rsplit('/', 1)
        if len(sp_result_text) == 2:
            token, label = result_text.rsplit('/', 1)
            if len(token) == 0:
                token = ' '

            if result_text == ' /O':
                list_result.append( (token, label) )
            else:
                token = token[1:]
                if len(token) == 0:
                    token = ' '
                list_result.append( (token, label) )
    return list_result

def convertSeq2(in_seqtext):
    list_result = []
    pivot_idx = 0
    for e in re.finditer('<[^<]+:[A-Z|a-z|-]+>', in_seqtext):
        start_idx, end_idx = e.span()
        while pivot_idx < start_idx:
            list_result.append( (in_seqtext[pivot_idx], 'O') )
            pivot_idx += 1

        sp_token = e.group()[1:-1].rsplit(':', 1)
        if len(sp_token) == 2:
            token, label = sp_token
            for idx, token_e in enumerate(token):
                if len(token_e) == 0:
                    token_e = ' '
                if idx == 0:
                    list_result.append((token_e, f'B-{label}'))
                else:
                    list_result.append((token_e, f'I-{label}'))

        pivot_idx = end_idx

    while pivot_idx < len(in_seqtext):
        token_e = in_seqtext[pivot_idx]
        if len(token_e) == 0:
            token_e = ' '
        list_result.append( (token_e, 'O') )
        pivot_idx += 1
    
    return list_result


def main():
    args = parse_argument()
    
    list_instance = []
    with open(args.pred_fpath, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr.readlines()):
            if idx == 0: continue
            json_obj = json.loads(line.replace('\n',''))
            for k, v in json_obj.items():
                list_instance.append( (k, v[-1]) )
    
    fw = open(args.output_fpath, 'w', encoding='utf-8')
    if args.task == 'classification':
        list_BUFF = []
        for _id, _result in list_instance:
            dict_result = {_id:_result}
            json_buff = json.dumps(dict_result, ensure_ascii=False)
            list_BUFF.append(json_buff)
        fw.write('\n'.join(list_BUFF))
        
    elif args.task == 'sequencetagging':
        list_BUFF = []
        json_text = json.dumps({'id':[('token','label'), ('token','label'), ('token','label')]})
        list_BUFF.append(json_text)
        for _id, _result in list_instance:
            list_result = convertSeq(_result)
            dict_result = {_id:list_result}
            json_buff = json.dumps(dict_result, ensure_ascii=False)
            list_BUFF.append(json_buff)
        fw.write('\n'.join(list_BUFF))
        
    elif args.task == 'sequencetagging-2':
        list_BUFF = []
        json_text = json.dumps({'id':[('token','label'), ('token','label'), ('token','label')]})
        list_BUFF.append(json_text)
        for _id, _result in list_instance:
            list_result = convertSeq2(_result)
            dict_result = {_id:list_result}
            json_buff = json.dumps(dict_result, ensure_ascii=False)
            list_BUFF.append(json_buff)
        fw.write('\n'.join(list_BUFF))
        
    fw.close()


if __name__ == "__main__":
    main()
    