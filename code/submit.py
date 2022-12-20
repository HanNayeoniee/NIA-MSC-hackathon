#!/usr/bin/env python
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from train import seed_everything
import jsonlines
from tqdm import tqdm
import pandas as pd

start_time = time.time()

MODEL_NAME = "../model/1216_shuffle"
# MODEL_NAME = "../model/1219_memory"
CONFIG = AutoConfig.from_pretrained(MODEL_NAME)
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
MODEL_ET5 = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    from_tf=".ckpt" in MODEL_NAME,
    config=CONFIG,
)

print(f"Load Model Time: {time.time() - start_time}")


def predict_utterace(sent):
    """
    문자열 1개씩 인퍼런스
    """
    global MODEL_ET5, TOKENIZER
    

    MODEL_ET5.cuda()
    MODEL_ET5.eval()

    sent = "<s>" + sent
    tokenized_sent = TOKENIZER(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )

    
    bad_words_ids = list({"<user>": 45100, "</agent_memory>": 45105, "</user_memory>": 45103, "<dialogue>": 45107, \
        "<agent_memory>": 45104, "<user_memory>": 45102, "</dialogue>": 45108, "<agent>": 45101, "<empty>": 45106}.values())
    
    params = {
        'min_length': 20,
        'max_length': 300,
        'num_beams': 5,
        'early_stopping': True,
        'no_repeat_ngram_size': 2,
        'num_return_sequences': 1,
        'do_sample': True,
        'top_k': 0,
        'temperature': 0.5,
        'top_p': 1.0,
        "bad_words_ids": bad_words_ids
    }
    # print(params['num_beams'])
    # print(type(params['num_beams']))

    
    with torch.no_grad():
        outputs = MODEL_ET5.generate(
            input_ids=tokenized_sent['input_ids'].cuda(),
            attention_mask=tokenized_sent['attention_mask'].cuda(),
            min_length=params['min_length'],
            max_length=params['max_length'],
            # num_beams=params['num_beams'],
            early_stopping=params['early_stopping'],
            no_repeat_ngram_size=params['no_repeat_ngram_size'],
            num_return_sequences=params['num_return_sequences'],
            do_sample=params['do_sample'],
            top_k=params['top_k'],
            temperature=params['temperature'],
            top_p=params['top_p'],
            bad_words_ids=[params['bad_words_ids']]
            )

        

    pred_res = TOKENIZER.decode(outputs[0])
    pred_res = pred_res.replace("<pad><dialogue> 생성:<agent>", "").replace("</dialogue></s>", "").strip()
    pred_res = pred_res.split("</dialogue>")[0]

    return pred_res



if __name__ == "__main__":
    seed_everything(42)
    
    # MODEL_NAME = "../model/1216_shuffle"
    # sent = "<memory><empty></memory><dialogue>입력:<user>시간이 참 빨리 가네요... 네 잘 지냈어요. 애들 시험은 잘 봤나요?<agent>안녕하세요~! 시험 준비하느라 바쁘게 지내다보니 벌써 한 달이나 지났어요~ 잘 지내셨나요?<user><empty><agent><empty><user><empty><agent><empty></dialogue>"
    # pred = predict_utterace(MODEL_NAME, sent)
    # print("prediction:", pred)
    
    
    data_path = "../dataset/nia_dataset/valid_memory2.jsonl"
    
    out_df = pd.DataFrame()
    with jsonlines.open(data_path) as f:
        for i, line in tqdm(enumerate(f.iter())):    
            # if i == 100:
            #     break

            index = list(line.keys())[0]
            seq = list(line.values())[0]
            in_seq = seq[0]
            if in_seq == "in_seq":
                continue
            
            pred = predict_utterace(in_seq)
            out_df = out_df.append({
                "pred": pred,
            }, ignore_index=True)
            
    out_df.to_excel("./memory2_output_me2.xlsx")
            
