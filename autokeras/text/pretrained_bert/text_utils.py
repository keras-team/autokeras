import torch


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """ Convert text examples to BERT specific input format.

    Tokenize the input text and convert into features.

    Args:
        examples: Text data.
        tokenizer: Tokenizer to process the text into tokens.
        max_seq_length: The maximum length of the text sequence supported.

    Returns:
        all_input_ids: ndarray containing the ids for each token.
        all_input_masks: ndarray containing 1's or 0's based on if the tokens are real or padded.
        all_segment_ids: ndarray containing all 0's since it is a classification task.
    """
    features = []
    for (_, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        if len(input_ids) != max_seq_length or \
                len(input_mask) != max_seq_length or \
                len(segment_ids) != max_seq_length:
            raise AssertionError()

        features.append(InputFeatures(input_ids=input_ids,
                                      input_mask=input_mask,
                                      segment_ids=segment_ids))

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    return all_input_ids, all_input_mask, all_segment_ids
