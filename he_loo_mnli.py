import csv
from tqdm import tqdm
import os, copy, json, dataclasses
import pickle
from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer


@dataclass
class InputExample:
    guid: str
    text_a: str
    text_b: Optional[str] = None
    label: Optional[str] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self), indent=2) + "\n"


@dataclass(frozen=True)
class InputFeatures:
    input_ids: List[int]
    attention_mask: Optional[List[int]] = None
    token_type_ids: Optional[List[int]] = None
    label: Optional[Union[int, float]] = None

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def _glue_convert_examples_to_features(
        examples: List[InputExample],
        tokenizer: PreTrainedTokenizer,
        max_length: Optional[int] = None,
        label_map=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    labels = [label_map[example.label] for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    return features


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def _create_examples(lines, set_type):
    """Creates examples for the training, dev and test sets."""

    examples = []
    for (i, line) in enumerate(lines):
        if i == 0:
            continue
        guid = "%s-%s" % (set_type, line[0])
        text_a = line[8]
        text_b = line[9]
        label = None if set_type.startswith("test") else line[-1]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def text_pre(text_data, model, tokenizer, max_seq_length=128):
    # Switch the model to eval mode.
    model.eval()
    label_list = ["contradiction", "entailment", "neutral"]
    label_map = {label: i for i, label in enumerate(label_list)}

    features = _glue_convert_examples_to_features(text_data, tokenizer, max_seq_length, label_map)
    input_ids = torch.tensor([feature.input_ids for feature in features]).cuda()
    attention_mask = torch.tensor([feature.attention_mask for feature in features]).cuda()
    token_type_ids = torch.tensor([feature.token_type_ids for feature in features]).cuda()
    labels = torch.tensor([feature.label for feature in features]).cuda()

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    logits = logits[1]
    logits = nn.functional.softmax(logits, dim=-1)

    return logits


def process_stop_word(x):
    punctions = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}']
    text = x.text_a
    text = text.split(" ")
    text_temp = text[0]
    for word in text[1:]:
        if word not in punctions and word[0] != "'":
            text_temp = text_temp + " " + word
        else:
            text_temp = text_temp + word

    y = copy.deepcopy(x)
    y.text_a = text_temp
    return y


def del_list_by_v(_list, v):
    while v in _list:
        _list.remove(v)
    return _list


def Cal_pre(id, pre_label, _example_ori, _words_ori_a, _words_ori_b, model, tokenizer):
    if id == None:
        _example_temp = copy.deepcopy(_example_ori)
        out = [_example_temp]
    else:
        out = []
        for idd in id:
            words_temp_a = []
            words_temp_b = []
            _example_temp = copy.deepcopy(_example_ori)

            for i in range(len(_words_ori_a)):
                if i not in idd:
                    words_temp_a.append(_words_ori_a[i])
                else:
                    words_temp_a.append('[PAD]')
            words_temp_a = ' '.join(words_temp_a)

            for i in range(len(_words_ori_b)):
                if i + len(_words_ori_a) + 1 not in idd:
                    words_temp_b.append(_words_ori_b[i])
                else:
                    words_temp_b.append('[PAD]')
            words_temp_b = ' '.join(words_temp_b)

            _example_temp.text_a = words_temp_a
            _example_temp.text_b = words_temp_b

            out.append(_example_temp)

    result = text_pre(out, model, tokenizer)
    if pre_label == None:
        pre_label = torch.argmax(result).cpu().item()
        result = result[0][pre_label].cpu().item()
        return result, pre_label
    else:
        result = [x[pre_label].cpu().item() for x in result]
        return result

def Cal_score(id, pre_label, _example_ori, words_ori_a, words_ori_b, model, tokenizer):
    if len(id) <= 50:
        id1 = [x[0] for x in id]
        id2 = []
        for item in id:
            x, y = item
            assert x != y
            z = copy.deepcopy(x)
            z.extend(y)
            id2.append(z)
        pres1 = Cal_pre(id1, pre_label, _example_ori, words_ori_a, words_ori_b, model, tokenizer)
        pres2 = Cal_pre(id2, pre_label, _example_ori, words_ori_a, words_ori_b, model, tokenizer)
        scores = [x - y for x, y in zip(pres1, pres2)]
        return scores
    else:
        scores_all = []
        iter_l = 100
        left = len(id) % iter_l

        iter_num = int(len(id) / iter_l) + 1 if left != 0 else int(len(id) / iter_l)
        for i in range(iter_num):
            s = iter_l * i
            e = s + iter_l
            id1 = [x[0] for x in id[s:e]]
            id2 = []
            for item in id[s:e]:
                x, y = item
                assert x != y
                z = copy.deepcopy(x)
                z.extend(y)
                id2.append(z)
            pres1 = Cal_pre(id1, pre_label, _example_ori, words_ori_a, words_ori_b, model, tokenizer)
            pres2 = Cal_pre(id2, pre_label, _example_ori, words_ori_a, words_ori_b, model, tokenizer)
            scores = [x - y for x, y in zip(pres1, pres2)]
            scores_all.extend(scores)
        assert len(scores_all) == len(id)
        return scores_all


if __name__ == '__main__':
    pretrained_dir = ''  # model path
    data_dir = ''  # data path
    out_dir = ''  # output file

    dev_data_matched = _create_examples(_read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
    dev_data_mismatched = _create_examples(_read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")
    dev_data = []
    dev_data.extend(dev_data_matched[0:500])
    dev_data.extend(dev_data_mismatched[0:500])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir).cuda()
    model.eval()

    output = []
    for example_ori in tqdm(dev_data[:]):
        print(example_ori.text_a, example_ori.text_b)

        words_ori_a = example_ori.text_a.split(' ')
        words_ori_b = example_ori.text_b.split(' ')

        words_ori_a = del_list_by_v(words_ori_a, '')
        words_ori_b = del_list_by_v(words_ori_b, '')

        words_len_a = len(words_ori_a)
        words_len_b = len(words_ori_b)

        pre_ori, pre_label = Cal_pre(None, None, example_ori, words_ori_a, words_ori_b, model, tokenizer)

        indexes = list(range(words_len_a + words_len_b + 1))
        indexes = [[x] for x in indexes]

        new_pair = None
        indexes_list = []
        score_list = []

        score_record = {}

        while len(indexes) > 0:
            indexes_list.append(indexes)

            cal_score_list = []
            for x in indexes:
                cal_score_list.append([[], x])
                for y in indexes:
                    if x[0] != y[0]:
                        key = str([x, y])
                        if key not in score_record.keys():
                            cal_score_list.append([x, y])

            scores = Cal_score(cal_score_list, pre_label, example_ori, words_ori_a, words_ori_b, model, tokenizer)

            for cal_score, score in zip(cal_score_list, scores):
                score_record[str(cal_score)] = score

            if len(indexes) == 2:
                break

            candidate = []
            for x in indexes:
                item1 = [[], x]
                for y in indexes:
                    if x[0] < y[0] and x[0] != [words_len_a] and y[0] != [words_len_a]:
                        item2 = [y, x]
                        pair = [item1, item2]
                        score = abs(score_record[str(item1)] - score_record[str(item2)])
                        candidate.append([pair, score])
            candidate.sort(key=lambda x: x[1], reverse=1)

            max_pair, max_score = candidate[0]

            y = []
            y.extend(max_pair[1][0])
            y.extend(max_pair[1][1])
            y.sort()

            indexes2 = []
            already_add = 0
            for x in indexes:
                if x not in max_pair[1]:
                    indexes2.append(x)
                else:
                    if not already_add:
                        indexes2.append(y)
                        already_add = 1
            indexes = indexes2

        scores = []
        for indexes in indexes_list:
            score = []
            for x in indexes:
                key = [[], x]
                score.append(score_record[str(key)])
            scores.append(score)

        out = [indexes_list, scores, (words_ori_a, words_ori_b)]
        output.append(out)
        print('finished')

    with open(out_dir, 'wb') as f:
        pickle.dump(output, f)
