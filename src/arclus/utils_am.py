import logging
import os
import random
from logging import Logger
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from mlflow import log_param
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer
from transformers.data.processors import DataProcessor, InputExample, InputFeatures

logger = logging.getLogger(__name__)


def load_bert_model_and_data_no_args(
    model_path: str,
    task_name: str,
    batch_size: int,
    data_dir: str,
    overwrite_cache: bool,
    max_seq_length: int,
    model_type: str,
    cache_root: str,
) -> Tuple[DataLoader, TensorDataset, BertForSequenceClassification, List[Any]]:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    bert_model = BertForSequenceClassification.from_pretrained(model_path)
    data, examples = load_and_cache_examples(
        task=task_name,
        tokenizer=tokenizer,
        model_path=model_path,
        data_dir=data_dir,
        overwrite_cache=overwrite_cache,
        max_seq_length=max_seq_length,
        model_type=model_type,
        cache_root=cache_root,
    )
    sampler = SequentialSampler(data)
    guids = [o.guid for o in examples]
    return DataLoader(data, sampler=sampler, batch_size=batch_size), data, bert_model, guids


@torch.no_grad()
def inference_no_args(
    data: TensorDataset,
    loader: DataLoader,
    logger: Logger,
    model: BertForSequenceClassification,
    batch_size: int,
    softmax: bool = False,
) -> List[float]:
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    predictions = []
    logger.info("***** Running inference {} *****".format(""))
    logger.info("  Num examples = %d", len(data))
    logger.info("  Batch size = %d", batch_size)
    model.to(device)
    model.eval()
    for batch in tqdm(loader, desc="Inference"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2], }
        # logits = outputs[0].cpu().numpy()[:, 1]
        outputs = model(**inputs)
        logits = outputs[0]
        if softmax:
            logits = logits.log_softmax(dim=-1)
        logits = logits[:, 1]
        predictions.extend(logits.tolist())
    return predictions


@torch.no_grad()
def inference(args, data, loader, logger, model):
    return inference_no_args(
        data=data,
        loader=loader,
        logger=logger,
        model=model,
        batch_size=args.batch_size,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def load_and_cache_examples(
    task: str,
    tokenizer: BertTokenizer,
    model_path: str,
    data_dir: str,
    overwrite_cache: bool,
    max_seq_length: int,
    model_type: str,
    cache_root: str = "../../data/preprocessed",
):
    processor = processors[task]()
    output_mode = output_modes[task]
    # if args.active_learning:

    # Load data features from cache or dataset file
    # if active learning, the train data will be saved inside each learning iteration directory
    cached_features_file = os.path.join(cache_root, "cached_{}_{}_{}".format("inference", list(
        filter(None, model_path.split("/"))).pop(), str(task), ), )

    if os.path.exists(cached_features_file) and not overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = None
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        examples = processor.get_examples(data_dir=data_dir)
        log_param("  Num examples training", len(examples))

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(model_type in ["xlnet"]),
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.as_tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.as_tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.as_tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )
    #    if output_mode == "classification":
    #        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    #    elif output_mode == "regression":
    #        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids
    )
    return dataset, examples


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet
        where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    if task is not None:
        processor = processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert (len(attention_mask) == max_length), "Error with input length {} vs {}".format(len(attention_mask),
                                                                                              max_length)
        assert (len(token_type_ids) == max_length), "Error with input length {} vs {}".format(len(token_type_ids),
                                                                                              max_length)

        if output_mode == "classification":
            if example.label is None:
                label = None
            else:
                label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "attention_mask: %s" % " ".join([str(x) for x in attention_mask])
            )
            logger.info(
                "token_type_ids: %s" % " ".join([str(x) for x in token_type_ids])
            )
        #            logger.info("label: %s (id = %d)" % (example.label, label))
        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    return features


class SimilarityProcessor(DataProcessor):
    """Processor for the AM data set."""

    def get_examples(self, data_dir: str):
        df = self.read_tsv(data_dir)
        return self._create_examples(df)

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return ["unsimilar", "similar"]

    @staticmethod
    def read_tsv(input_file):
        return pd.read_csv(input_file, sep=";")

    @staticmethod
    def _create_examples(df):
        """Creates examples for the training and test sets."""
        examples = []
        for index, row in df.iterrows():
            guid = row["premise_id"], row["claim_id"]
            text_a = row["premise_text"]
            text_b = row["claim_text"]

            if type(text_a) is not str:
                text_a = ""
            if type(text_b) is not str:
                text_b = ""
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b)
            )
        return examples


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, task_name):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    if task_name == "SD" or task_name == "AM2":
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        return {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "acc_and_f1": (acc + f1_micro) / 2,
            "true_positive": tp,
            "true negative": tn,
            "false positive": fp,
            "false negative": fn
        }

    if task_name == "AM3" or task_name == "SIM":
        return {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "acc_and_f1": (acc + f1_micro) / 2,
        }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "SIM":
        return acc_and_f1(preds, labels, task_name)
    else:
        raise KeyError(task_name)


tasks_num_labels = {
    "SD": 2,
}
processors = {
    "SIM": SimilarityProcessor,
}
output_modes = {
    "SIM": "classification",
}
