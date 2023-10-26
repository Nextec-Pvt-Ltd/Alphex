from dataclasses import dataclass

@dataclass 
class Config: 
    n_token: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency 
    n_layer: int = 48 
    n_head: int = 12 
    d_model: int = 1600 
    d_head: int = 64 # d_model / n_head
    d_inner: int = 3072 # 4 * d_model 
    dropout: float = 0.2
    dropatt: float = 0.1 # dropout for attention weights
    tie_weight: bool = True # tie the word embedding and softmax weights 
    d_embed: int = None # dimension of the embeddings
    div_val: int = 1 # divide the embedding size by this value
    tie_projs: list = [False] # tie the projections in the adaptive softmax
    pre_lnorm: bool = False # apply LayerNorm to the input instead of the outpu
    tgt_len: int = 2 # target length
    ext_len: int = 2 # extended length for memory 
    mem_len: int = 5 # length of the memory for Transformer-XL 
    cutoffs: list = [] # cutoffs for the adaptive softmax 
    adapt_inp: bool = False # use adaptive input embeddings
    same_length: bool = False # use the same attn length for all tokens 
    attn_type: int = 0 # attention type. 0 for Transformer-XL, 1 for learnable embeddings, 2 for absolute embeddings, 3 for relative embeddings. 
    clamp_len: int = -1 # clamp all relative distances larger than clamp_len. -1 means no clamping. 
    sample_softmax: int = -1 # number of samples in sampled softmax