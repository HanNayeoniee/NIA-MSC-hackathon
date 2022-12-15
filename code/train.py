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

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Union, List

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    MBartTokenizerFast,
    set_seed,
)
from transformers.trainer_utils import EvaluationStrategy, is_main_process
from transformers.training_args import ParallelMode
from seq2seq.seq2seq_trainer import Seq2SeqTrainer
from seq2seq.seq2seq_training_args import Seq2SeqTrainingArguments
from seq2seq.utils import (
    Seq2SeqDataCollator,
    assert_all_frozen,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)
########## kmh
from arguments import ModelArguments, DataTrainingArguments
from seq2seq_data_processor import YNAT_Processor, one_txtcl_Processor, two_txtcl_Processor, seq2seq_Processor, seq2seq_jsonl_Processor
from seq2seq_dataset_et5 import Seq2SeqDataset_ET5
from seq2seq_data_utils import build_compute_metrics_fn_et5
##########


logger = logging.getLogger(__name__)

def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 3 and sys.argv[-1].endswith(".json"):  ### jihee
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))  ### jihee
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    training_args.output_dir = data_args.output_dpath
    model_args.model_name_or_path = data_args.pretrained_model

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['<USER>', '<AGENT>', '<KEYWORD>', '<GUIDE>', '<STYLE>', '<EMPTY>', '<DIAL>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    print("#" * 30)
    print("Len tokenzier ==> ", len(tokenizer))
    print(tokenizer.tokenize("안녕하세요 <USER> <AGENT> <KEYWORD> <GUIDE> <STYLE> <EMPTY> <DIAL>"))
    print()
    print()


    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=".ckpt" in model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    ########## custom - token resize
    model.resize_token_embeddings(len(tokenizer))
    print(model)


    

    ########## custom - YNAT
    if data_args.process_model == 'ynat':
        processor = YNAT_Processor()
    elif data_args.process_model == 'onetxtcl':
        processor = one_txtcl_Processor()
    elif data_args.process_model == 'twotxtcl':
        processor = two_txtcl_Processor()
    elif data_args.process_model == 'seq2seq':
        processor = seq2seq_Processor()
    elif data_args.process_model == 'seq2seq_jsonl':
        processor = seq2seq_jsonl_Processor()
        
    dataset_class = Seq2SeqDataset_ET5

    if training_args.predict_with_generate:
        compute_metrics_fn = build_compute_metrics_fn_et5(tokenizer)
    else:
        compute_metrics_fn = None

    if training_args.do_predict == False and training_args.predict_with_generate:
        compute_metrics_fn = None
    ##########



    # use task specific params
    use_task_specific_params(model, data_args.task)

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.tgt_lang)

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    sp_train_path = data_args.train_fpath.split('/')
    train_dpath = '/'.join(sp_train_path[:-1])
    train_type_path, train_file_type  = sp_train_path[-1].split('.')
    
    # Get datasets
    train_dataset = (
        dataset_class(
            tokenizer,
            type_path=train_dpath + '/' + train_type_path,
            data_dir=data_args.data_dir,
            processor=processor,        
            n_obs=data_args.n_train,
            max_target_length=data_args.max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            filetype=train_file_type
        )
        if training_args.do_train
        else None
    )
    
    sp_dev_path = data_args.dev_fpath.split('/')
    dev_dpath = '/'.join(sp_dev_path[:-1])
    dev_type_path, dev_file_type  = sp_dev_path[-1].split('.')
    
    eval_dataset = (
        dataset_class(
            tokenizer,
            type_path=dev_type_path,
            data_dir=dev_dpath,
            processor=processor,      
            n_obs=data_args.n_val,
            max_target_length=data_args.val_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            filetype=dev_file_type
        )
        if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
        else None
    )    
    
    test_dataset = (
        dataset_class(
            tokenizer,
            type_path=dev_type_path,
            data_dir=dev_dpath,
            processor=processor,       
            n_obs=data_args.n_test,
            max_target_length=data_args.test_max_target_length,
            max_source_length=data_args.max_source_length,
            prefix=model.config.prefix or "",
            filetype=dev_file_type
        )
        if training_args.do_predict or training_args.predict_with_generate
        else None
    )
    print()
    print()
    print("GOOD1")
    print()
    print()

    print('[*]', len(test_dataset))
    print('[**]', test_dataset[0])
    print('[***]', test_dataset[-1])

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=Seq2SeqDataCollator(
            tokenizer, data_args, model.config.decoder_start_token_id, training_args.tpu_num_cores
        ),
        compute_metrics=compute_metrics_fn,
        tokenizer=tokenizer,
        
    )

    
    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        #train_result = trainer.train(
        #    model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        #)
        train_result = trainer.train()

        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        
        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)

            # 0916 추가
            tokenizer.save_pretrained(training_args.output_dir)
            # tokenizer.save_pretrained(os.path.join(training_args.output_dir, "tokenizer_model.json"))

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(metric_key_prefix="val")
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                test_preds = tokenizer.batch_decode(
                    test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))
    else:
        print('[]', sum([1 for e in test_dataset]) )
        print('[*]', len(test_dataset))
        print('[*]', test_dataset[-3:])
        
        test_output = trainer.predict(test_dataset=test_dataset, metric_key_prefix="test")
        
        print('[**]', len(test_output))
        print('[**]', test_output[-3:])
        
        if training_args.predict_with_generate:
            test_preds = tokenizer.batch_decode(
                test_output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            print('[***]', len(test_preds))
            print('[***]', test_preds[-3:])
                
            test_preds = lmap(str.strip, test_preds)
            write_txt_file(test_preds, os.path.join(training_args.output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(training_args.output_dir, "all_results.json"))
        logger.info(all_metrics)

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
