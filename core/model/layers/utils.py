# util functions

import torch
import torch.nn as nn


def masked_softmax(X, valid_len):
    """
    masked softmax for attention scores
    args:
        X: 3-D tensor, valid_len: 1-D or 2-D tensor
    """
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(
                valid_len, repeats=shape[1], dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = X.reshape(-1, shape[-1])
        for count, row in enumerate(X):
            row[int(valid_len[count]):] = -1e6
        return nn.functional.softmax(X.reshape(shape), dim=-1)
