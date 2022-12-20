# NIA-MSC-hackathon ("내일은 칼퇴근" 팀)



- 사용한 pre-trained model: **ETRI-t5-base**

- 최종 훈련된 모델 경로: `/workspace/model/1216_shuffle`

- 학습 로그: `/workspace/code/script/train.log`



## Docker

```bash
tar -zxvf nia_image.tar.gz

docker load -i nia_image.tar

docker run -it --network host --gpus all --name nia_image nia_image:latest /bin/bash

```



## Train



### 방법 1. train.py 실행

```bash

export CUDA_VISIBLE_DEVICES=0,1,2 # (필수) gpu 번호 지정

export CUDA_DEVICE_ORDER=PCI_BUS_ID



# pre-trained model

PRETRAIN_MODEL=/workspace/model/etri_t5



DATASET_DIR=/workspace/data

TRAIN_PATH=$DATASET_DIR/train_memory2.jsonl

DEV_PATH=$DATASET_DIR/valid_memory2.jsonl



# (필수) 학습 모델 저장 경로 지정

TRAIN_OUTPUT_PATH=/workspace/model/{학습 모델 저장 경로}/



mkdir $TRAIN_OUTPUT_PATH



nohup python -u train.py \

--pretrained_model $PRETRAIN_MODEL \

--train_fpath $TRAIN_PATH \

--dev_fpath $DEV_PATH \

--output_fpath $TRAIN_OUTPUT_PATH \

\

--do_train \

--model_name_or_path $PRETRAIN_MODEL \

--save_steps 10000 \

--per_device_train_batch_size 12 \

--gradient_accumulation_steps 1 \

--num_train_epochs 3.0 \

--n_gpu 3 \  # (선책) 사용할 gpu 개수 설정

--parallel_mode ParallelMode.NOT_DISTRIBUTED \

--train_dataloader random \

--do_predict \

--predict_with_generate 1> {학습 로그 경로} 2>&1 & # (필수) 학습 로그 경로 지정

```

### 방법 2. train.sh 실행

```bash

cd script

nohup ./train.sh 1> {학습 로그 경로} 2>&1 & # (필수) 학습 로그 경로 지정

```




## Inference

```bash

cd code

python generate_chat.py --input_folder ./input --output_folder ./output

```





## Code Structure

```bash

.

├── code

│   ├── arguments.py

│   ├── generate_chat.py

│   ├── input

│   ├── output

│   ├── runs

│   ├── script

│   ├── seq2seq

│   ├── seq2seq_data_processor.py

│   ├── seq2seq_dataset_et5.py

│   ├── seq2seq_data_utils.py

│   ├── submit.py

│   └── train.py

├── data

│   ├── train.jsonl

│   └── valid.jsonl

├── dockerfile

├── model

│   ├── 1216_shuffle

│   ├── etri_t5

│   └── pbert_iq_ns

├── README.md

└── requirements.txt

```



## 활용 모델

### ET5-base

pre-trained model로 ETRI에서 공개한 ET5-base 모델을 사용하였다.

- URL: https://aiopen.etri.re.kr/et5Model

- 모델 세부 내용

    - 학습데이터: 136GB 원시 말뭉치

    - 딥러닝 라이브러리: pytorch

    - HuggingFace model 및 SentencePiece tokenizer model 파일

    - Latin alphabets: Cased

- 모델 파라미터

    - 45100 vocabs

    - 12 layers (인코더/디코더 각각)

    - 12 heads

    - 768 d_model

    - 64 d_kv

    -3072 d_ff



### PBERT

- 개요

    - 문장 임베딩 모델(SBERT-IQ)를 응용하여, Passage와 Sentence 간 유사도를 비교하는 모델

      (SBERT-IQ: "SBERT-IQ: 키워드 정보량을 고려한 Sentence-BERT 기반의 임베딩 모델", 솔트룩스, KCC 2022)

    - Dialogue Context의 3-turn과 Memory의 Sentence 간 유사도 비교를 통해, 현재 대화와 관련있는 Memory 검색

    - Generator(ET5)에 발화생성에 참고할 Memory Span 제공




## 멀티세션 대화 데이터 피드백 내용

- 다음 대화 시작이 "16시간만이네요.", "3일만이네요."와 같은 패턴으로 시작되는 것이 부자연스러워 보였습니다.
