# NIA-MSC-hackathon


- 사용한 pre-trained model: ETRI-t5-base

## Train
```bash
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

###### 경로 수정
PRETRAIN_MODEL=../model/etri_t5

DATASET_DIR=../dataset/nia_dataset/
TRAIN_PATH=$DATASET_DIR/train_memory2.jsonl
DEV_PATH=$DATASET_DIR/valid_memory2.jsonl


###### 경로 수정
TRAIN_OUTPUT_PATH=../model/1219_memory/ 

mkdir $TRAIN_OUTPUT_PATH


cd ..
pwd

python train.py \
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
--n_gpu 3 \
--parallel_mode ParallelMode.NOT_DISTRIBUTED \
--train_dataloader random \
--do_predict \
--predict_with_generate
```


## Inference
```bash
python generate_chat.py --input_folder ./input --output_folder ./output
```



## Code Structure
```bash
et5
├── assets
├── data
├── code
│   ├── script
│   │   ├── train.sh
│   │   └── infer.sh
│   ├── seq2seq/
│   ├── arguments.py
│   ├── convert_result.py
│   ├── eval.py
│   ├── generate_chat.py
│   ├── infer.py
│   ├── seq2seq_data_processor.py
│   ├── seq2seq_data_utils.py
│   ├── seq2seq_dataset_et5.py
│   ├── submit.py
│   └── train.py
├── dataset
│   └── nia_dataset/
├── model
│   ├── etri_t5/
│   └── 버전별 모델
├── README.md
└── requirements.txt
```