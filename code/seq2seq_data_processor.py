import json

from tqdm import tqdm
from transformers.data.processors.utils import DataProcessor
from seq2seq_data_utils import read_json



class YNAT_Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the input sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class YNAT_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            src_texts.append("YNAT sentence: " + e.text)
            tgt_texts.append(e.label)

        return (src_texts, tgt_texts)


    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        # lines = self._read_tsv(filename)    # read_tsv(filename)
        print(f"#####\t Reading an input file ...\t {filename}")
        lines = read_json(filename)
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line["title"]
            label = line["label"]
            examples.append(
                YNAT_Example(guid=guid, text=text, label=label))
    
        return examples


class one_txtcl_Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the input sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label
    
class one_txtcl_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            src_texts.append("ONE 문장 1: " + e.text)
            tgt_texts.append(e.label)
        
        
        return (src_texts, tgt_texts)


    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        print(f"#####\t Reading an input file ...\t {filename}")
        lines = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                lines.append(line.strip())                                
                
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text = line[-2]
            label = line[-1]
            examples.append(
                one_txtcl_Example(guid=guid, text=text, label=label))
    
        return examples


class two_txtcl_Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text1, text2, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the input sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text1 = text1
        self.text2 = text2
        self.label = label
    
class two_txtcl_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            src_texts.append("TWO 문장1: " + e.text1 + " 문장2: " + e.text2)
            tgt_texts.append(e.label)
        
        for i in range(3):
            print('[*]', src_texts[i])
            print('[**]', tgt_texts[i])

        return (src_texts, tgt_texts)


    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        print(f"#####\t Reading an input file ...\t {filename}")
        lines = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr.readlines():
                lines.append(line.strip())                                
                
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            sp_line = line.split('\t')
            
            text1 = sp_line[-3]
            text2 = sp_line[-2]
            label = sp_line[-1]
            
            examples.append(
                two_txtcl_Example(guid=guid, text1=text1, text2=text2, label=label))
    
        return examples

    
class seq2seq_Example(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, inseq, outseq=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the input sequence.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.inseq = inseq
        self.outseq = outseq
    
class seq2seq_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        list_id = []
        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            list_id.append(e.guid)
            src_texts.append("SEQ2SEQ 실행: " + e.inseq)
            tgt_texts.append(e.outseq)
            
#         src_texts.append("NONE")
#         tgt_texts.append("NONE")
        
        
        return (list_id, src_texts, tgt_texts)


    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        print(f"#####\t Reading an input file ...\t {filename}")
        lines = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr.readlines()):                
                lines.append(line.strip())                                
                
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                print(line)
                continue
            guid = "%s-%s" % (set_type, i)
            sp_line = line.split('\t')
            inseq = sp_line[-2]
            outseq = sp_line[-1]
            examples.append(
                seq2seq_Example(guid=sp_line[0], inseq=inseq, outseq=outseq))
    
        return examples
    
    
class seq2seq_jsonl_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        list_id = []
        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            list_id.append(e.guid)
            src_texts.append(e.inseq)
            tgt_texts.append(e.outseq)
        
        return (list_id, src_texts, tgt_texts)

    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        print(f"#####\t Reading an input file ...\t {filename}")
        lines = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr.readlines()):                
                lines.append(line.strip())                                
                
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                print(line)
                continue
            json_obj = json.loads(line)
            for k,v in json_obj.items():
                examples.append(seq2seq_Example(guid=k, inseq=v[0], outseq=v[1]))
                
#             guid = "%s-%s" % (set_type, i)
#             sp_line = line.split('\t')
#             inseq = sp_line[-2]
#             outseq = sp_line[-1]
#             examples.append(
#                 seq2seq_Example(guid=sp_line[0], inseq=inseq, outseq=outseq))
    
        return examples
    
    

class seq2seq_jsonl_BART_Processor(DataProcessor):
    """
    Processor for the YNAT(KLUE) data set.
    """

    def get_seq2seq_examples(self, filename, set_type="train"):
        examples = self.get_examples(filename, set_type)

        list_id = []
        src_texts = []
        tgt_texts = []
        for e in tqdm(examples, desc="#####\t Get source and target texts ... "):
            list_id.append(e.guid)
            src_texts.append(e.inseq + '</s>')
            tgt_texts.append(e.outseq + '</s>')
        
        return (list_id, src_texts, tgt_texts)

    def get_examples(self, filename, set_type="train"):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """

        print(f"#####\t Reading an input file ...\t {filename}")
        lines = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for idx, line in enumerate(fr.readlines()):                
                lines.append(line.strip())                                
                
        examples = self.create_examples(lines, set_type)

        return examples


    def create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc="#####\t Create examples ... "):
            if i == 0:
                print(line)
                continue
            json_obj = json.loads(line)
            for k,v in json_obj.items():
                examples.append(seq2seq_Example(guid=k, inseq=v[0], outseq=v[1]))
                
#             guid = "%s-%s" % (set_type, i)
#             sp_line = line.split('\t')
#             inseq = sp_line[-2]
#             outseq = sp_line[-1]
#             examples.append(
#                 seq2seq_Example(guid=sp_line[0], inseq=inseq, outseq=outseq))
    
        return examples