from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from seqeval.scheme import IOB2
import json
from types import SimpleNamespace
import re


class Token:
    def __init__(self, text=None, pred_tag=None, gold_tag=None):
        """
        Token object to hold token attributes
        :param text: str
        :param pred_tag: str
        :param gold_tag: str
        """
        self.text = text
        self.gold_tag = gold_tag
        self.pred_tag = pred_tag

    def __str__(self):
        """
        Token text representation
        :return: str
        """
        gold_tags = "|".join(self.gold_tag)

        if self.pred_tag:
            pred_tags = "|".join([pred_tag["tag"] for pred_tag in self.pred_tag])
        else:
            pred_tags = ""

        if self.gold_tag:
            r = f"{self.text}\t{gold_tags}\t{pred_tags}"
        else:
            r = f"{self.text}\t{pred_tags}"

        return r
    

class TagVocab:
    def __init__(self):
        self.vocab = None
        nested_index_filename = "IdToTag-nested.json"

        with open(nested_index_filename, "r") as fh:
            self.vocab = json.load(fh)

    def get_itos(self):
        vocabs = list()

        for v in self.vocab:
            vocab = {int(k): v for k, v in v.items()}
            vocab = dict(sorted(vocab.items()))
            vocabs.append(list(vocab.values()))

        return vocabs
    

def conll_to_segments(filename, tags_start_col=1):
    """
    Convert CoNLL files to segments. This return list of segments and each segment is
    a list of tuples (token, tag)
    :param filename: Path
    :return: list[[tuple]] - [[(token, tag), (token, tag), ...], [(token, tag), ...]]
    """
    segments, segment = list(), list()

    with open(filename, "r") as fh:
        for token in fh.read().splitlines():
            if not token.strip():
                segments.append(segment)
                segment = list()
            else:
                parts = token.split()
                token = Token(text=parts[0], gold_tag=parts[tags_start_col:])
                segment.append(token)

        segments.append(segment)

    return segments


def compute_nested_metrics(segments):
    """
    Compute metrics for nested NER
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    vocabs = TagVocab().get_itos()
    y, y_hat = list(), list()

    # We duplicate the dataset N times, where N is the number of entity types
    # For each copy, we create y and y_hat
    # Example: first copy, will create pairs of ground truth and predicted labels for entity type GPE
    #          another copy will create pairs for LOC, etc.
    for i, vocab in enumerate(vocabs):
        vocab_tags = [tag for tag in vocab if "-" in tag]
        r = re.compile("|".join(vocab_tags))

        y += [[(list(filter(r.match, token.gold_tag)) or ["O"])[0] for token in segment] for segment in segments]
        y_hat += [[token.pred_tag[i]["tag"] for token in segment] for segment in segments]

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro", scheme=IOB2),
        "macro_f1": f1_score(y, y_hat, average="macro", scheme=IOB2),
        "weights_f1": f1_score(y, y_hat, average="weighted", scheme=IOB2),
        "precision": precision_score(y, y_hat, scheme=IOB2),
        "recall": recall_score(y, y_hat, scheme=IOB2),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)


def compute_single_label_metrics(segments):
    """
    Compute metrics for flat NER
    :param segments: List[List[arabiner.data.dataset.Token]] - list of segments
    :return: metrics - SimpleNamespace - F1/micro/macro/weights, recall, precision, accuracy
    """
    y = [[token.gold_tag[0] for token in segment] for segment in segments]
    y_hat = [[token.pred_tag[0]["tag"] for token in segment] for segment in segments]

    metrics = {
        "micro_f1": f1_score(y, y_hat, average="micro", scheme=IOB2),
        "macro_f1": f1_score(y, y_hat, average="macro", scheme=IOB2),
        "weights_f1": f1_score(y, y_hat, average="weighted", scheme=IOB2),
        "precision": precision_score(y, y_hat, scheme=IOB2),
        "recall": recall_score(y, y_hat, scheme=IOB2),
        "accuracy": accuracy_score(y, y_hat),
    }

    return SimpleNamespace(**metrics)


if __name__ == "__main__":
    print("Testing")
    vocabs = TagVocab().get_itos()
    truth_segments = conll_to_segments("test.txt", tags_start_col=1)
    pred_segments = conll_to_segments("predictions.txt", tags_start_col=2)

    # Our Github repo saves predictions pipe (|) separated and adds a header
    # So, we will check if the format is proper ConLL format or not and transform
    # it if needed, if the first line starts with "Token", then there is a header
    if pred_segments[0][0].text == "Token":
        pred_segments[0].pop(0)
    if truth_segments[0][0].text == "Token":
        truth_segments[0].pop(0)

    # Count number of columns in the gold_tag attribute, if it is 2, then 
    # it is likely it is pipe separated, then tranform the data
    if len(pred_segments[0][0].gold_tag) == 1:
        for segment in pred_segments:
            for token in segment:
                tags = token.gold_tag[0].split("|")
                token.gold_tag = tags

    # truth segments is List[List[Token]] and each Token has pref_tag attribute
    # Now we want to assign the predicted tag in the pred_file to the Tokens in the truth_file
    # This requires that both files are 100% aligned, one extra line will cause problems
    # and inaccurate results

    # We need to valiadate the user input and make sure it matches the ground truth data

    # First, is this nested or flat NER, we can tell from the number of tags in the first token of the first segment
    nested = True if len(pred_segments[0][0].gold_tag) > 1 else False
    line = 1

    # Second, make sure the number of segments match between ghround truth and predicted file
    assert len(truth_segments) == len(pred_segments), "Number of truth and predicted segment mismatch ({} != {})".format(len(truth_segments), len(pred_segments))

    # Third, validate each token in eahc segment
    for truth_segment, pred_segment in zip(truth_segments, pred_segments):

        # Are the number of tokens in each segment equal
        #truth_tokens = [t.text for t in truth_segment]
        #pred_tokens = [t.text for t in pred_segment]
        assert len(truth_segment) == len(pred_segment), "Number of tokens in truth and pedicted segment mismatch at line {}".format(line)

        # Validate the token pair in ground truth and predicted  
        for tt, pt in zip(truth_segment, pred_segment):

            # Are the tokens (words) the same
            assert tt.text == pt.text, "mismatch in tokens at line {} ({} != {})".format(line, tt.text, pt.text)

            # Does each token in the predictions has the correct number of tags?
            assert ((len(pt.gold_tag) == len(vocabs) - 1) and nested) or (len(pt.gold_tag) == 1 and not nested), "mismatch in number of tags at line {}".format(line)

            tt.pred_tag = [{"tag": t} for t in pt.gold_tag] 

            line += 1

        line += 1

    if nested:
        # Compute nested NER metrics  
        metrics = compute_nested_metrics(truth_segments)
    else:
        # Compute flat NER metrics
        metrics = compute_single_label_metrics(truth_segments)
            

    print(json.dumps(metrics.__dict__, indent=4))
     

