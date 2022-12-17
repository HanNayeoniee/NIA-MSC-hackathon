export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID

###### 경로 수정
PRETRAIN_MODEL=../model/etri_t5

DATASET_DIR=../dataset/nia_dataset/
TRAIN_PATH=$DATASET_DIR/train.jsonl
DEV_PATH=$DATASET_DIR/valid.jsonl


###### 경로 수정
TRAIN_OUTPUT_PATH=../model/1216_shuffle/ 

# mkdir $TRAIN_OUTPUT_PATH


cd ..
pwd

python train.py \
--pretrained_model $PRETRAIN_MODEL \
--train_fpath $TRAIN_PATH \
--dev_fpath $DEV_PATH \
--output_dpath $TRAIN_OUTPUT_PATH \
\
--do_train \
--model_name_or_path $PRETRAIN_MODEL \
--save_steps 10000 \
--per_device_train_batch_size 12 \
--gradient_accumulation_steps 1 \
--num_train_epochs 3.0 \
--n_gpu 3 \
--parallel_mode ParallelMode.NOT_DISTRIBUTED \
--do_predict \
--predict_with_generate
# popd