import logging
import os
import pickle
import time

import torch
from torch.utils.data.dataset import Dataset

from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm

#from tokenization_utils import BatchEncoding


from trainer import torch_distributed_zero_first

from typing import Callable, Dict, List, NamedTuple, Optional, Tuple



logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False, local_rank=-1,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)

def batch_encode_plus(lines,tokenizer, \
        add_special_tokens: bool = True, \
        max_length: Optional[int] = None, \
        stride: int = 0, \
        doc_stride: int = 0, \
        truncation_strategy: str = "longest_first",\
        pad_to_max_length: bool = False,\
        is_pretokenized: bool = False,\
        return_tensors: Optional[str] = None,\
        return_token_type_ids: Optional[bool] = None,\
        return_attention_masks: Optional[bool] = None,\
        return_overflowing_tokens: bool = False,\
        return_special_tokens_masks: bool = False,\
        return_offsets_mapping: bool = False,\
        return_lengths: bool = False,\
        **kwargs):


    batch_outputs = {}

    for line in tqdm(lines):
        
        line = line.strip()
        words = line.split()
        
        word_tokens = []
        char_tokens = []
        
        for word in words:
            tokens = tokenizer.tokenize(word)
            for i,token in enumerate(tokens):
                if token.startswith("##") or i==0:
                    word_tokens.append(1)
                else:
                    word_tokens.append(2) 

                char_tokens.append(token)
        
        ids = tokenizer.convert_tokens_to_ids(char_tokens)        
   
        span_ids = ids

        span_word_tokens = word_tokens

        while True:

            if span_ids == []:
                break

            if len(span_ids) > max_length-2:
                _ids = span_ids[:max_length-2]
                span_ids = span_ids[doc_stride:]
                _word_tokens = span_word_tokens[:max_length-2]
                span_word_tokens = span_word_tokens[doc_stride:]
            else:
                _ids = span_ids
                span_ids = []
                _word_tokens = span_word_tokens
                

            outputs = tokenizer.prepare_for_model(
                    _ids,
                    pair_ids=None,
                    max_length=max_length,
                    pad_to_max_length=pad_to_max_length,
                    add_special_tokens=add_special_tokens,
                    stride=stride,
                    truncation_strategy=truncation_strategy,
                    return_attention_mask=return_attention_masks,
                    return_token_type_ids=return_token_type_ids,
                    return_overflowing_tokens=return_overflowing_tokens,
                    return_special_tokens_mask=return_special_tokens_masks,
                    #return_lengths=return_lengths,
                    return_tensors=None,  # We will convert the whole batch to tensors at the end
                )

            _word_tokens = [0] + _word_tokens + [0]
            assert len(_word_tokens) == len(outputs["input_ids"])


            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)
            if "words_token" not in batch_outputs: 
                batch_outputs["words_token"] = []
            batch_outputs["words_token"].append(_word_tokens)


    if return_tensors is not None:
        tokenizer.convert_to_tensors_(batch_outputs, return_tensors)

#    return BatchEncoding(batch_outputs)    
    return batch_outputs

class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1,doc_stride=128):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = batch_encode_plus(lines, tokenizer, add_special_tokens=True, max_length=block_size,doc_stride=doc_stride)
        
        self.examples = (batch_encoding["input_ids"],batch_encoding["words_token"])
        #self.examples = batch_encoding

    def __len__(self):
        return len(self.examples[0])


    def __getitem__(self, i) -> torch.Tensor:
        
        #return torch.tensor(self.examples[i], dtype=torch.long)
        return (torch.tensor(self.examples[0][i]),torch.tensor(self.examples[1][i]))
