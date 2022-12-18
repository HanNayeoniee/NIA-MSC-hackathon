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



def predict_utterace(MODEL_NAME, sent):
    """
    문자열 1개씩 인퍼런스
    """
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        from_tf=".ckpt" in MODEL_NAME,
        config=config,
    )

    # # set num_beams for evaluation
    # if data_args.eval_beams is None:
    #     data_args.eval_beams = model.config.num_beams


    model.cuda()
    model.eval()

    tokenized_sent = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    
    
    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=tokenized_sent['input_ids'].cuda(),
            attention_mask=tokenized_sent['attention_mask'].cuda(),
            max_length=200
            )

    pred_res = tokenizer.decode(outputs[0])
    pred_res = pred_res.replace("<pad><dialogue> 생성:<agent>", "").replace("</dialogue></s>", "").strip()

    end_time = time.time()
    print('predict time', end_time - start_time)

    return pred_res



if __name__ == "__main__":
    seed_everything(42)
    
    MODEL_NAME = "../model/1216_shuffle"
    sent = "<memory><empty></memory><dialogue>입력:<user>시간이 참 빨리 가네요... 네 잘 지냈어요. 애들 시험은 잘 봤나요?<agent>안녕하세요~! 시험 준비하느라 바쁘게 지내다보니 벌써 한 달이나 지났어요~ 잘 지내셨나요?<user><empty><agent><empty><user><empty><agent><empty></dialogue>"

    pred = predict_utterace(MODEL_NAME, sent)
    print("prediction:", pred)
