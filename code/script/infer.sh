export GIT_PYTHON_REFRESH=quiet
export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_DEVICE_ORDER=PCI_BUS_ID 

###### 경로 수정
MODEL_PATH=../model/1216_shuffle
DATASET_DIR=../dataset/nia_dataset
TEST_PATH=$DATASET_DIR/valid.jsonl
INFER_OUTPUT_PATH=$DATASET_DIR/infer_1216_shuffle.jsonl


# mkdir $TRAIN_OUTPUT_PATH

cd ../
pwd


# INPUT : 학습완료모델, 테스트데이터
# OUTPUT : 추론결과
python infer.py \
--pretrained_model $MODEL_PATH \
--test_fpath $TEST_PATH \
--output_fpath $INFER_OUTPUT_PATH \
\
--process_model seq2seq_jsonl \
--model_name_or_path '' \
--data_dir '' \
--per_device_eval_batch_size 12 \
--predict_with_generate
# popd