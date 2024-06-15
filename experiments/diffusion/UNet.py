import math

import torch


def sinusoidal_embedding(vocab_len, inner_dim):
    embeddings = []
    for vocab_pos in range(vocab_len):
        cur_embed = []
        for embed_ind in range(inner_dim):
          val = vocab_pos / 10000 ** (2 * embed_ind / inner_dim)
          if embed_ind % 2 == 0:
              val = math.sin(val)
          else:
              val = math.cos(val)
          cur_embed.append(val)
        embeddings.append(torch.tensor(cur_embed))
    embeddings = torch.stack(embeddings)
    return embeddings

