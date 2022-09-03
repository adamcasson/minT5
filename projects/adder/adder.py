from typing import Literal

import torch
from torch.utils.data import Dataset


class AdditionDataset(Dataset):
    """
    Modified version of karpathy/minGPT's AdditionDataset for encoder/decoder model

    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the model:
    "8550531"
    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      model, because of how the addition algorithm works.
    As one more example, the problem 6 + 39 = 45 would be encoded as:
    "0639" for the encoder, and "x05" for the decoder, and "045" for the targets.
    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits
    to the encoder, and "x" to the decoder and hoping that the model completes the
    sequence with the next (n+1) digits correctly.
    """

    def __init__(self, ndigit: int, split: Literal['train', 'test']):
        self.ndigit = ndigit
        self.split = split  # train/test

        # split up all addition problems into either training data or test data
        assert (
            self.ndigit <= 3
        ), "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (10**self.ndigit) ** 2  # total number of possible addition problems with ndigit numbers
        rng = torch.Generator()
        rng.manual_seed(1337)
        perm = torch.randperm(num, generator=rng)
        num_test = min(int(num * 0.8), 500)  # 20% of the whole dataset, or only up to 500
        self.ixes = perm[:num_test] if split == 'test' else perm[num_test:]

    def get_vocab_size(self):
        return 10 + 1  # digits 0..9 plus pad token

    def get_context_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        # return 3 * self.ndigit + 1 - 1
        return 2 * self.ndigit

    def __len__(self):
        return self.ixes.nelement()

    def __getitem__(self, idx):
        ndigit = self.ndigit
        # given a problem index idx, first recover the associated a + b
        idx = self.ixes[idx].item()
        nd = 10**ndigit
        a = idx // nd
        b = idx % nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{ndigit}d' % a
        bstr = f'%0{ndigit}d' % b
        cstr = (f'%0{ndigit+1}d' % c)[::-1]  # reverse c to make addition easier
        render = astr + bstr
        dix_src = [int(s) for s in render]  # convert each character to its token index
        dix_dst = [10] + [int(s) for s in cstr]  # prepend BOS token for decoder input
        # x will be input to the encoder and y will be the decoder input/labels
        src_idx = torch.tensor(dix_src, dtype=torch.long)
        dst_idx = torch.tensor(dix_dst[:-1], dtype=torch.long)  # don't include EOS/last token
        targets = torch.tensor(dix_dst[1:], dtype=torch.long)  # don't include BOS token in target
        return src_idx, dst_idx, targets
