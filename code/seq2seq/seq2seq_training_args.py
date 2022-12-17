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
from dataclasses import dataclass, field
from typing import Optional

from seq2seq_trainer import arg_to_scheduler
from transformers import TrainingArguments


logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Parameters:
        label_smoothing (:obj:`float`, `optional`, defaults to 0):
            The label smoothing epsilon to apply (if not zero).
        sortish_sampler (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to SortishSamler or not. It sorts the inputs according to lenghts in-order to minimizing the padding size.
        predict_with_generate (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
    """

    label_smoothing: Optional[float] = field(
        default=0.0, 
        metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    sortish_sampler: bool = field(
        default=False, 
        metadata={"help": "Whether to SortishSamler or not."}
    )
    predict_with_generate: bool = field(
        default=False, 
        metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(
        default=False, 
        metadata={"help": "whether to use adafactor"}
    )
    encoder_layerdrop: Optional[float] = field(
        default=None, 
        metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, 
        metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(
        default=None, 
        metadata={"help": "Dropout probability. Goes into model.config."}
    )
    attention_dropout: Optional[float] = field(
        default=None, 
        metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "Output directory"}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "If `True`, overwrite the content of the output directory. \
                    Use this to continue training if `output_dir` points to a checkpoint directory."},
    )

    # multi-gpu 학습을 위한 파라미터 추가
    n_gpu: Optional[int] = field(
        default=1,
        metadata={"help": "The number of GPUs used by this process."},
    )
    parallel_mode: [str] = field(
        default="ParallelMode.NOT_PARALLEL",
        metadata={"help": "The current mode used for parallelism if multiple GPUs/TPU cores are available. \
                    (ParallelMode.NOT_DISTRIBUTED for several GPUs in one single process which uses torch.nn.DataParallel)"}
        # ParallelMode.NOT_DISTRIBUTED
    )

    # data를 순차적으로(sequential) 로드하기 위한 파라미터 추가
    train_dataloader: str = field(
        default="random",
        metadata={"help": "The current mode used to load train data. (sequential-SequentialSampler, random-RandomSampler)"}
    )
