import csv
from tqdm import tqdm
import os, copy, json, dataclasses
import numpy as np
import pickle
from typing import List, Optional, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizer
from lime.lime_text import LimeTextExplainer


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


class model_Pred():
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    def tokenfun(self, text):
        label_list = ["contradiction", "entailment", "neutral"]
        label_map = {label: i for i, label in enumerate(label_list)}
        text_a, text_b = text.split('[SEP]')

        text_data = InputExample(guid='guid', text_a=text_a, text_b=text_b, label='contradiction')
        features = _glue_convert_examples_to_features([text_data], tokenizer, self.max_length, label_map)

        input_ids = torch.tensor([feature.input_ids for feature in features]).cuda()
        attention_mask = torch.tensor([feature.attention_mask for feature in features]).cuda()
        token_type_ids = torch.tensor([feature.token_type_ids for feature in features]).cuda()
        labels = torch.tensor([feature.label for feature in features]).cuda()
        return input_ids, attention_mask, token_type_ids, labels

    def get_proba_list(self, tlist, indexes):
        label_list = ["contradiction", "entailment", "neutral"]
        label_map = {label: i for i, label in enumerate(label_list)}
        words = tlist[0].split(' ')

        l_a = 0
        l_b = 0
        flag = 0
        for word in words:
            if word == '[SEP]':
                flag = 1
            else:
                if flag == 0:
                    l_a += 1
                else:
                    l_b += 1
        tlist2 = []

        for x in tlist:
            xx = x.replace("UNKWORDZ", "[PAD]")
            tlist2.append(xx)

        tlist2 = [text.split(' ') for text in tlist2]

        for t in tlist2:
            if t[l_a] == "[PAD]":
                t[l_a] = "[SEP]"
            assert t[l_a] == "[SEP]"

        text_data = [
            InputExample(guid='guid', text_a=' '.join(x[0:l_a]), text_b=' '.join(x[l_a + 1:]), label='contradiction')
            for x in tlist2]

        features = _glue_convert_examples_to_features(text_data, tokenizer, self.max_length, label_map)
        input_ids = torch.tensor([feature.input_ids for feature in features]).cuda()
        attention_mask = torch.tensor([feature.attention_mask for feature in features]).cuda()
        token_type_ids = torch.tensor([feature.token_type_ids for feature in features]).cuda()
        labels = torch.tensor([feature.label for feature in features]).cuda()

        batch_size = 500
        epoch = int(input_ids.size()[0] / batch_size)

        for i in range(epoch):
            s = i * batch_size
            e = s + batch_size

            input_ids_temp = input_ids[s:e]
            attention_mask_temp = attention_mask[s:e]
            token_type_ids_temp = token_type_ids[s:e]
            labels_temp = labels[s:e]

            with torch.no_grad():
                logits_temp = model(
                    input_ids=input_ids_temp,
                    attention_mask=attention_mask_temp,
                    token_type_ids=token_type_ids_temp,
                    labels=labels_temp,
                )
            logits_temp = logits_temp[1]
            if i == 0:
                logits = logits_temp
            else:
                logits = torch.cat((logits, logits_temp), 0)

        logits = nn.functional.softmax(logits, dim=-1)

        return logits.cpu().numpy()


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


def cut(text):
    out = text.split(' ')
    out = del_list_by_v(out, '')
    return out


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
                if i + len(_words_ori_a + 1) not in idd:
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


def Cal_score(id, pre_label, _example_ori, _words_ori_a, _words_ori_b, model, tokenizer, indexes=None):
    text = []

    for i, word in enumerate(_words_ori_a):
        if i not in id:
            text.append(word)
        else:
            text.append('[PAD]')

    text.append('[SEP]')
    for i, word in enumerate(_words_ori_b):
        if i + len(_words_ori_a) + 1 not in id:
            text.append(word)
        else:
            text.append('[PAD]')

    text = ' '.join(text)
    exps = explainer.explain_instance(text, mobj.get_proba_list, labels=[pre_label], num_samples=2000, num_features=128,
                                      indexes=indexes, sep_id=len(_words_ori_a))

    exp_dicts = [exp.local_exp[pre_label] for exp in exps]

    outs = []
    for exp_dict in exp_dicts:
        if not outs:
            out = [None for x in range(len(_words_ori_a) + len(_words_ori_b) + 1)]
        else:
            out = outs[0][:]

        for item in exp_dict:
            _, val = item
            out[_] = val

        assert None not in out
        outs.append(out)
    return outs


def convert_indexes_score(x, _indexes):
    out = []
    for ids in _indexes:
        val = 0
        for id in ids:
            val += x[id]
        out.append(val)
    assert len(out) == len(_indexes)
    return out


if __name__ == '__main__':
    pretrained_dir = ''  # model path
    data_dir = ''  # data path
    out_dir = ''  # output file

    dev_data_matched = _create_examples(_read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")
    dev_data_mismatched = _create_examples(_read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_mismatched")
    dev_data = []
    dev_data.extend(dev_data_matched[0:250])
    dev_data.extend(dev_data_mismatched[0:250])
    dev_data.extend(dev_data_matched[250:500])
    dev_data.extend(dev_data_mismatched[250:500])

    tokenizer = AutoTokenizer.from_pretrained(pretrained_dir)
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_dir).cuda()
    model.eval()
    mobj = model_Pred(model, tokenizer)

    class_names = {"contradiction", "entailment", "neutral"}
    explainer = LimeTextExplainer(class_names=class_names, split_expression=cut, bow=False)

    output = []

    for example_ori in tqdm(dev_data[:1000]):
        print(example_ori.text_a, example_ori.text_b)

        words_ori_a = example_ori.text_a.split(' ')
        words_ori_b = example_ori.text_b.split(' ')
        words_ori_a = del_list_by_v(words_ori_a, '')
        words_ori_b = del_list_by_v(words_ori_b, '')

        words_len_a = len(words_ori_a)
        words_len_b = len(words_ori_b)

        if words_len_a + words_len_b > 125:
            continue

        pre_ori, pre_label = Cal_pre(None, None, example_ori, words_ori_a, words_ori_b, model, tokenizer)

        indexes = list(range(words_len_a + words_len_b + 1))
        indexes = [[x] for x in indexes]

        new_pair = None
        indexes_list = []
        score_list = []

        del_record = {}
        iter_num = 0
        while len(indexes) > 0 and iter_num < 10:
            iter_num += 1

            indexes_list.append(indexes)

            if len(indexes) == 1:
                break

            scores = []
            indexes2 = [[]]
            indexes2.extend(indexes)
            outs = Cal_score([], pre_label, example_ori, words_ori_a, words_ori_b, model, tokenizer, indexes=indexes2)

            for index, out in zip(indexes2, outs):
                key = str(index)
                del_record[key] = out
                if key != '[]':
                    scores.append(del_record[key])

            scores = [convert_indexes_score(x, indexes) for x in scores]
            scores = np.array(scores)
            scores_base = del_record['[]']

            scores_base = convert_indexes_score(scores_base, indexes)
            scores_base = np.array(scores_base)

            for _ in range(len(scores)):
                scores[_][_] = scores_base[_]

            score_list.append(scores_base)
            scores_offset = scores - scores_base

            for i in range(len(scores_offset)):
                for j in range(len(scores_offset)):
                    scores_offset[i][j], scores_offset[j][i] = scores_offset[i][j] + scores_offset[j][i], 0

            max_id = []
            max_val = -10000000

            for x in range(len(indexes)):
                for y in range(len(indexes)):
                    val = abs(scores_offset[x][y])
                    if x != y:
                        if val > max_val:
                            max_val = val
                            max_id = [x, y]

            max_pair = [indexes[max_id[0]], indexes[max_id[1]]]

            y = []
            for item in max_pair:
                y.extend(item)
                y.sort()

            indexes2 = []
            already_add = 0
            for x in indexes:
                if x not in max_pair:
                    indexes2.append(x)
                else:
                    if not already_add:
                        indexes2.append(y)
                        already_add = 1
            indexes = indexes2
            print(indexes)

        scores_all = score_list
        out = [indexes_list, scores_all, (words_ori_a, words_ori_b)]
        output.append(out)

    with open(out_dir, 'wb') as f:
        pickle.dump(output, f)
