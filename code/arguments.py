from dataclasses import dataclass, field
from typing import Optional, Union, List

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(
        default=False, 
        metadata={"help": "Whether tp freeze the encoder."}
    )
    freeze_embeds: bool = field(
        default=False, 
        metadata={"help": "Whether  to freeze the embeddings."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    process_model: Optional[str] = field(
        default="seq2seq_jsonl",
        metadata={"help": "mode of data processing"}
    )
    # data_name: str = field(
    #     # default="klue_ner",
    #     metadata={"help": "data name"}
    # )
    # file_type: str = field(   
    #     default = None,     
    #     metadata={"help": "data name"}
    # )
    data_dir: str = field(
        default = "",
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={"help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=142,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(
        default=-1, 
        metadata={"help": "# training examples. -1 means use all."}
    )
    n_val: Optional[int] = field(
        default=-1, 
        metadata={"help": "# validation examples. -1 means use all."}
    )
    n_test: Optional[int] = field(
        default=-1, 
        metadata={"help": "# test examples. -1 means use all."}
    )
    src_lang: Optional[str] = field(
        default=None, 
        metadata={"help": "Source language id for translation."}
    )
    tgt_lang: Optional[str] = field(
        default=None, 
        metadata={"help": "Target language id for translation."}
    )
    eval_beams: Optional[int] = field(
        default=None, 
        metadata={"help": "# num_beams to use for evaluation."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
        
    train_fpath: Optional[str] = field(
        default=None, 
        metadata={"help": "Target language id for translation."}
    )
    dev_fpath: Optional[str] = field(
        default=None, 
        metadata={"help": "Target language id for translation."}
    )
    output_dpath: Optional[str] = field(
        default=None, 
        metadata={"help": "Target language id for translation."}
    )
    pretrained_model: Optional[str] = field(
        default=None, 
        metadata={"help": "Target language id for translation."}
    )