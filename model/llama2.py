import torch
import torch.nn as nn
import torch.nn.functional as F
from model.basemodel import BaseModel
from module.layers import SeqPoolingLayer
from module import data_augmentation
from data import dataset
from copy import deepcopy

import abc
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn

from model.ndp_module import NDPModule
from model.similarity_module import GeneralizedInteractionModule
from model.embedding_modules import EmbeddingModule
from model.utils import get_current_embeddings
from model.input_features_preprocessors import InputFeaturesPreprocessorModule
from model.output_postprocessors import OutputPostprocessorModule
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    # vocab_size: int = -1  # defined later by tokenizer -> In recommender system, it replaces by itemnum
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    maxlen: int = 50



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    #print(xq.shape,xk.shape,freqs_cos.shape,freqs_sin.shape)
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class LLaMA2(BaseModel):
    def __init__(self, config, dataset_list : list[dataset.BaseDataset]) -> None:
        super().__init__(config, dataset_list)
        self.query_encoder = SASRecQueryEncoder(
            self.max_seq_len,
            1,
            self.embed_dim,
            config['model']['layer_num'],
            config['model']['head_num'],
            config['model']['hidden_size'],
            config['model']['activation'],
            config['model']['dropout_rate'],



        )
'''
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    # vocab_size: int = -1  # defined later by tokenizer -> In recommender system, it replaces by itemnum
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0
    maxlen: int = 50
'''
class LLaMA2QueryEncoder(GeneralizedInteractionModule):
    """
    Implements SASRec (Self-Attentive Sequential Recommendation, https://arxiv.org/abs/1808.09781, ICDM'18).

    Compared with the original paper which used BCE loss, this implementation is modified so that
    we can utilize a Sampled Softmax loss proposed in Revisiting Neural Retrieval on Accelerators
    (https://arxiv.org/abs/2306.04039, KDD'23) and Turning Dross Into Gold Loss: is BERT4Rec really
    better than SASRec? (https://arxiv.org/abs/2309.07602, RecSys'23), where the authors showed
    sampled softmax loss to significantly improved SASRec model quality.
    """

    def __init__(
        self,
        max_sequence_len: int,
        max_output_len: int,
        embedding_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        ffn_activation_fn: str,
        ffn_dropout_rate: float,
        embedding_module: EmbeddingModule,
        similarity_module: NDPModule,
        input_features_preproc_module: InputFeaturesPreprocessorModule,
        output_postproc_module: OutputPostprocessorModule,
        dropout_rate: float,
        activation_checkpoint: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(ndp_module=similarity_module)
        self._embedding_module: EmbeddingModule = embedding_module
        self._embedding_dim: int = embedding_dim
        self._item_embedding_dim: int = embedding_module.item_embedding_dim
        self._max_sequence_length: int = max_sequence_len + max_output_len
        self._input_features_preproc: InputFeaturesPreprocessorModule = input_features_preproc_module
        self._output_postproc: OutputPostprocessorModule = output_postproc_module
        self._activation_checkpoint: bool = activation_checkpoint
        self._verbose: bool = verbose

        #self.attention_layers = torch.nn.ModuleList()
        #self.forward_layers = torch.nn.ModuleList()
        self._num_blocks: int = num_blocks
        self._num_heads: int = num_heads
        self._ffn_hidden_dim: int = ffn_hidden_dim
        self._ffn_activation_fn: str = ffn_activation_fn
        self._ffn_dropout_rate: float = ffn_dropout_rate

        args_dict=dict(
            dim=embedding_dim,
            n_layers=num_blocks,
            n_heads=num_heads,
            n_kv_heads=num_heads,
            # vocab_size=64793, # defined later by tokenizer -> In recommender system, it replaces by itemnum
            multiple_of=32,
            norm_eps=1e-5,
            max_seq_len=max_sequence_len + max_output_len,
            dropout=0.0,
            maxlen=9999999 #no_use_at_all
        )
        params=ModelArgs(**args_dict)
        self.params=params
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.maxlen)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        #self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        #self.output = nn.Linear(args_dict['dim'], self.item_num+1, bias=False)
        
        '''
        for _ in range(num_blocks):
            self.attention_layers.append(
                torch.nn.MultiheadAttention(embed_dim=self._embedding_dim,
                                            num_heads=num_heads,
                                            dropout=ffn_dropout_rate,
                                            batch_first=True)
            )
            self.forward_layers.append(
                StandardAttentionFF(
                    embedding_dim=self._embedding_dim,
                    hidden_dim=ffn_hidden_dim,
                    activation_fn=ffn_activation_fn,
                    dropout_rate=self._ffn_dropout_rate,
                )
            )
        '''
        self.register_buffer(
            "_attn_mask",
            torch.triu(
                torch.ones((self._max_sequence_length, self._max_sequence_length), dtype=torch.bool),
                diagonal=1,
            )
        )
        self.reset_state()

    def reset_state(self) -> None:
        for name, params in self.named_parameters():
            if "_input_features_preproc" in name or "_embedding_module" in name or "_output_postproc" in name:
                if self._verbose:
                    print(f"Skipping initialization for {name}")
                continue
            try:
                torch.nn.init.xavier_normal_(params.data)
                if self._verbose:
                    print(f"Initialize {name} as xavier normal: {params.data.size()} params")
            except:
                if self._verbose:
                    print(f"Failed to initialize {name}: {params.data.size()} params")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding_module.get_item_embeddings(item_ids)

    def debug_str(self) -> str:
        return (
            f"SASRec-d{self._item_embedding_dim}-b{self._num_blocks}-h{self._num_heads}"
            + "-" + self._input_features_preproc.debug_str()
            + "-" + self._output_postproc.debug_str()
            + f"-ffn{self._ffn_hidden_dim}-{self._ffn_activation_fn}-d{self._ffn_dropout_rate}"
            + f"{'-ac' if self._activation_checkpoint else ''}"
        )
    '''
    def _run_one_layer(
        self,
        i: int,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        Q = F.layer_norm(
            user_embeddings, normalized_shape=(self._embedding_dim,), eps=1e-8,
        )
        mha_outputs, _ = self.attention_layers[i](
            query=Q,
            key=user_embeddings,
            value=user_embeddings,
            attn_mask=self._attn_mask,
        )
        user_embeddings = self.forward_layers[i](
            F.layer_norm(
                Q + mha_outputs,
                normalized_shape=(self._embedding_dim,),
                eps=1e-8,
            )
        )
        user_embeddings *= valid_mask
        return user_embeddings
    '''
    def _run_all_layers(
        self,
        user_embeddings: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batchsize, seqlen,hiddim = user_embeddings.shape
        user_embeddings *= self._embedding_dim ** 0.5
        # cite from LLaMA2
        h = self.dropout(user_embeddings)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
            h *= valid_mask
        
        #log_feats = self.norm(h)

        return h

    def generate_user_embeddings(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            past_ids: (B, N,) x int

        Returns:
            (B, N, D,) x float
        """
        past_lengths, user_embeddings, valid_mask = self._input_features_preproc(
            past_lengths=past_lengths,
            past_ids=past_ids,
            past_embeddings=past_embeddings,
            past_payloads=past_payloads,
        )
        user_embeddings=self._run_all_layers( user_embeddings, valid_mask)
        #replaced
        '''
        for i in range(len(self.attention_layers)):
            if self._activation_checkpoint:
                user_embeddings = torch.utils.checkpoint.checkpoint(
                    self._run_one_layer, i, user_embeddings, valid_mask,
                    use_reentrant=False,
                )
            else:
                user_embeddings = self._run_one_layer(i, user_embeddings, valid_mask)
        '''
        return self._output_postproc(user_embeddings)

    def forward(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
        batch_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            past_ids: [B, N] x int64 where the latest engaged ids come first. In
                particular, [:, 0] should correspond to the last engaged values.
            past_ratings: [B, N] x int64.
            past_timestamps: [B, N] x int64.

        Returns:
            encoded_embeddings of [B, N, D].
        """
        encoded_embeddings = self.generate_user_embeddings(
            past_lengths,
            past_ids,
            past_embeddings,
            past_payloads,
        )
        return encoded_embeddings

    def encode(
        self,
        past_lengths: torch.Tensor,
        past_ids: torch.Tensor,  # [B, N] x int64
        past_embeddings: torch.Tensor,
        past_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        encoded_seq_embeddings = self.generate_user_embeddings(past_lengths, past_ids, past_embeddings, past_payloads)  # [B, N, D]
        return get_current_embeddings(lengths=past_lengths, encoded_embeddings=encoded_seq_embeddings)

    def predict(
        self,
        past_ids: torch.Tensor,
        past_ratings: torch.Tensor,
        past_timestamps: torch.Tensor,
        next_timestamps: torch.Tensor,
        target_ids: torch.Tensor,
        batch_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.interaction(
            self.encode(past_ids, past_ratings, past_timestamps, next_timestamps),
            target_ids,
            batch_id=batch_id,
        )  # [B, X]