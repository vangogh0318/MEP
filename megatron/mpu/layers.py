# Copyright (c) 2021, EleutherAI contributors
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
from megatron import mpu

from .initialize import get_model_parallel_rank
from .initialize import get_model_parallel_world_size
from .mappings import copy_to_model_parallel_region
from .mappings import gather_from_model_parallel_region
from .mappings import reduce_from_model_parallel_region
from .mappings import scatter_to_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import VocabUtility


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    neox_args,
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=torch.float, requires_grad=False
    )
    init_method(master_weight)
    master_weight = master_weight.to(dtype=neox_args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(
        self, neox_args, num_embeddings, embedding_dim, init_method=init.xavier_normal_
    ):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocabulary dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings, get_model_parallel_rank(), self.model_parallel_size
        )
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=1
            )

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(
            masked_input,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ParallelKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None
    
    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        
        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias


class ParallelKerplePower(torch.nn.Module):
    """Kernelized Alibi Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.neox_args = neox_args
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        # add 
        self.input_dim = 1
        self.hidden_dim = 32 #similar to google FIRE paper parameters
        #self.output_dim = 1
        self.output_dim = self.num_heads_per_partition
        self.alibi_scaling = neox_args.alibi_scaling
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # bias_kernel = -bias_a*|m-n|^bias_p
        # weight_kernel = exp(-wei_a*|m-n|^wei_p)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_a, self.bias_p, self.wei_a, self.wei_p = None, None, None, None

        #if self.pos_emb.endswith('original'):
        if 'original' in self.pos_emb:
            print("power original")
            slopes = torch.Tensor(self._get_slopes(self.heads))[
                self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
            ][:,None,None]
            print('before')
            print(slopes)
            if self.alibi_scaling < 10:
                slopes = (2**self.alibi_scaling)*slopes
            else: 
                slopes = (self.alibi_scaling)*slopes
            print('after')
            print(slopes)

            slopes = slopes.to(torch.cuda.current_device()).to(neox_args.params_dtype)
            self.bias_a = Parameter(slopes, requires_grad=False)
        else:
            print("not original, in ap in pos_emb branch")
            bias_arg, wei_arg = self.pos_emb.split('_')[-2:]
            self.bias_p = get_parameter(2, 'uniform') if 'p' in bias_arg else None
            self.bias_a = get_parameter(1, 'uniform') if 'a' in bias_arg else None
            #self.wei_p = get_parameter(2, 'uniform') if 'p' in wei_arg else None
            #self.wei_a = get_parameter(1, 'uniform') if 'a' in wei_arg else None

        self.cached_matrix = None
        self.cached_seq_len = None

        if 'learned' in self.pos_emb:
            self.fc1 = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.gelu1 = torch.nn.GELU()
            self.fc2 = torch.nn.Linear(self.hidden_dim, self.output_dim)
    
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))
        if self.bias_p is not None:
            self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('bias_p', self.bias_p))
        if self.wei_a is not None:
            self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
            dd.update(get_stats('wei_a', self.wei_a))
        if self.wei_p is not None:
            self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
            dd.update(get_stats('wei_p', self.wei_p))
        return dd
    
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            #print("self.cached_seq_len != seq_len_k,%s,%s" % (self.cached_seq_len, seq_len_k))
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )

            # loss not best
            if 'clip' in self.pos_emb:
                print('power clip')
                #diff = torch.clip(diff, min=0, max=512)
                my_seq_len_k = 512
                diff = torch.where(diff < my_seq_len_k, diff, my_seq_len_k-1 + (diff-my_seq_len_k+1) / my_seq_len_k)

            #diff = diff.to(x.dtype)

            # Standardization (Z-score Normalization)
            # variance = diff.pow(2).mean()
            # epsilon = 1e-6
            # diff = diff * torch.rsqrt(variance + epsilon)
            #s_mean = cseq.to(torch.float32).mean()
            #diff = torch.tril((diff-s_mean) * torch.rsqrt(s_variance + epsilon))
            #diff = diff * torch.rsqrt(s_variance + epsilon)
            if 'norm' in self.pos_emb:
                print('power norm')
                # for test
                my_seq_len_k = seq_len_k
                if my_seq_len_k > 512:
                    my_seq_len_k = 512
                #if my_seq_len_k > 2**self.alibi_scaling :
                #    my_seq_len_k = 2**self.alibi_scaling

                cseq = torch.arange(my_seq_len_k)
                s_variance = cseq.to(torch.float32).pow(2).mean()
                s_mean = cseq.to(torch.float32).mean()
                epsilon = 1e-6
                #diff = diff * torch.rsqrt(s_variance + epsilon)
                diff = torch.tril((diff-s_mean) * torch.rsqrt(s_variance + epsilon))



            if 'learned' in self.pos_emb:
                #diff = -diff
                #for i in range(seq_len_q):
                #    print(diff[i])
                print('power learned')
                print("diff.view before shape")
                print((diff.shape))
                diff = diff.view(seq_len_q*seq_len_k, 1)
                print("diff.view(seq_len_q*seq_len_k, 1) after shape")
                print((diff.shape))
                diff = diff.to(x.dtype)
                diff = self.fc1(diff)
                print("self.fc1(diff) after shape")
                print((diff.shape))
                diff = self.gelu1(diff)
                print("self.gelu1(diff) after shape")
                print((diff.shape))
                diff = self.fc2(diff)
                print("self.fc2(diff) after shape")
                print((diff.shape))
                #diff = torch.tril(diff)
                #diff = diff.view(my_seq_len_q, my_seq_len_k)
                diff = diff.transpose(1,0).contiguous()
                print("diff.transpose(1,0).contiguous() after shape")
                print((diff.shape))
                diff = diff.view(diff.shape[0], seq_len_q, seq_len_k)
                print(" diff.view(diff.shape[0], seq_len_q, seq_len_k) after shape")
                print((diff.shape))
                diff = torch.tril(diff.to(torch.float32), diagonal=0)

            diff = diff.to(x.dtype)
            #print(diff)
            #print(diff.shape)

            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        # get bias matrix
        if self.bias_p is None and self.bias_a is None:
            #print("self.bias_p is None and self.bias_a is None")
            bias = 0.0
            bias = -diff
        else:
            if self.bias_p is not None:
                self.bias_p.data = self.bias_p.data.clamp(min=self.eps, max=2)
                bias = diff.pow(self.bias_p)
            else:
                bias = diff
            if self.bias_a is not None:
                self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
                bias = -bias*self.bias_a
            else:
                bias = -bias

        # get weight matrix
        if self.wei_p is None and self.wei_a is None:
            wei = 1.0
        else:
            if self.wei_p is not None:
                self.wei_p.data = self.wei_p.data.clamp(min=self.eps, max=2)
                wei = diff.pow(self.wei_p)
            else:
                wei = diff
            if self.wei_a is not None:
                self.wei_a.data = self.wei_a.data.clamp(min=self.eps)
                wei = (-wei*self.wei_a).exp()
            else:
                wei = (-wei).exp()
        
        '''
        print("myprint neox_args")
        print(self.neox_args)
        print("myprint neox_args")
        print("seq_len_k")
        print(self.cached_seq_len)
        print(seq_len_k)
        print(self.cached_seq_len)
        print(diff.shape)
        print(diff[seq_len_k-1])
        print(bias[0][seq_len_k-1])
        print("wei:%s" % (wei))
        '''

        if seq_len_q != seq_len_k:
            # print("mmmm neox_args")
            # print(self.neox_args)
            # print("mmmm neox_args")
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
            if type(wei) != float:
                wei = wei[:, seq_len_k - 1, :].view(wei.shape[0], 1, wei.shape[2])

        return x*wei + bias

class ParallelSNOPE_old(torch.nn.Module):
    """Scaling Norm Alibi Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.neox_args = neox_args
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        #for snope
        self.alibi_scaling = neox_args.alibi_scaling
        self.scale_factor_num = neox_args.scale_factor_num
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # bias_kernel = -alibi weight
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )

        self.bias_a = None
        self.bias_b = None

        print("snope original")
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ][:,None,None]
        print('before')
        print(slopes)

        if self.alibi_scaling < 10:
            slopes = (2**self.alibi_scaling)*slopes
        else:
            slopes = (self.alibi_scaling)*slopes
        print('after')
        print(slopes)

        slopes = slopes.to(torch.cuda.current_device()).to(neox_args.params_dtype)
        self.bias_a = Parameter(slopes, requires_grad=False)
        #self.bias_b = Parameter(torch.Tensor([512]))
        #self.bias_b = get_parameter(1, 'onlyone')

        self.cached_matrix = None
        self.cached_seq_len = None

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        if self.bias_b is not None:
            self.bias_b.data = self.bias_b.data.clamp(min=self.eps)
            dd.update(get_stats('bias_b', self.bias_b))

        return dd
 
    def _snope_get_mscale(self, scale=1):
        if scale <= 1:
            return 1.0
        return 0.1 * math.log(scale) + 1.0

    def _snope_get_scaling_alibi_matrix(self, diff, seq_len_k, is_norm=False):
        # loss not best
        #if 'clip' in self.pos_emb:
        #    print('power clip')
        #    #diff = torch.clip(diff, min=0, max=512)
        #    my_seq_len_k = 512
        #    method 1
        #    diff = torch.where(diff < my_seq_len_k, diff, my_seq_len_k-1 + (diff-my_seq_len_k+1) / my_seq_len_k)
        #    method 2
        #    diff = torch.where(diff < my_seq_len_k, diff, my_seq_len_k-1 + (diff-my_seq_len_k+1) / seq_len_k)
        #    method 4
        #    if seq_len_k > 1024
        #       svalue = 512 / seq_len_k
        #    else:
        #        svalue = 1
        #    diff = diff * svalue

        # method 3, not good
        #svalue = 512 / seq_len_k
        #diff = diff * svalue

        #clip
        #print(diff)
        #for norm
        print("self.scale_factor_num")
        print(self.scale_factor_num)
        print("self.scale_factor_num")
        if is_norm :
            #diff = torch.clip(diff, min=-2, max=2)
            #diff = torch.clip(diff, min=-0.8656, max=0.8656)
            #diff = torch.clip(diff, min=-1, max=1)
            diff = torch.clip(diff, min=-self.scale_factor_num, max=self.scale_factor_num)
        else:
            diff = torch.clip(diff, min=0, max=512)
        #print(diff)

        return diff

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            #print("self.cached_seq_len != seq_len_k,%s,%s" % (self.cached_seq_len, seq_len_k))
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            # before
            #diff = self._snope_get_scaling_alibi_matrix(diff, seq_len_k)

            # Standardization (Z-score Normalization)
            # s_mean = cseq.to(torch.float32).mean()
            # s_variance = diff.pow(2).mean()
            if 'norm' in self.pos_emb:
                print('snope norm')
                # for test
                my_seq_len_k = seq_len_k
                if my_seq_len_k > 512:
                    my_seq_len_k = 512

                cseq = torch.arange(my_seq_len_k)
                s_mean = cseq.to(torch.float32).mean()
                s_variance = cseq.to(torch.float32).pow(2).mean()
                epsilon = 1e-6

                diff = torch.tril((diff-s_mean) * torch.rsqrt(s_variance + epsilon))
                # after
                diff = self._snope_get_scaling_alibi_matrix(diff, seq_len_k, True)

            diff = diff.to(x.dtype)

            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        # get bias matrix
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        bias = diff
        bias = -bias*self.bias_a

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])
            if type(wei) != float:
                wei = wei[:, seq_len_k - 1, :].view(wei.shape[0], 1, wei.shape[2])

        return x + bias

###AliBi
#class AliBi(torch.nn.Module):
#copy from AliBi code
class ParallelAliBiLearning(torch.nn.Module):
#class ParallelSNOPE(torch.nn.Module):
    def __init__(self, neox_args):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        self.cached_matrix = None
        self.cached_seq_len = None

        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.eps = 1e-2

        self.alibi_scaling = neox_args.alibi_scaling

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_a = get_parameter(1, 'uniform')

        print("snope slopes:")
        print(self.bias_a)
        self.target_seq_len = 0

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        return dd
 
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        '''
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            diff = self.cached_matrix
        else:
            self.target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            diff = -torch.tril(
                torch.arange(self.target_seq_len)
                .view(self.target_seq_len, 1)
                .repeat(1, self.target_seq_len)
                + torch.arange(0, -self.target_seq_len, -1)
            )

        diff = diff.to(x.device).to(x.dtype)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        a = self.bias_a * diff

        self.cached_seq_len = self.target_seq_len
        self.cached_matrix = a
        '''

        if self.cached_seq_len != seq_len_k:
            diff = -torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        diff = diff.to(x.device).to(x.dtype)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        a = self.bias_a * diff
 
        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a

###AliBi
#class AliBi(torch.nn.Module):
#copy from AliBi code
class ParallelAliBi(torch.nn.Module):
#class ParallelSNOPE(torch.nn.Module):
    def __init__(self, neox_args):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        self.neox_args = neox_args
        self.heads = neox_args.num_attention_heads
        self.cached_matrix = None
        self.cached_seq_len = None

        self.neox_args = neox_args
        self.num_heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size

        self.alibi_scaling = neox_args.alibi_scaling
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ]
        print("snope alibi slopes:")
        print(slopes)
        print("self.alibi_scaling")
        print(self.alibi_scaling)

        self.register_buffer("slopes", slopes)

        self.bias_a = None

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        return dd
 
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            #0.return [start * ratio**i for i in range(n)]
            #1.return [1/(2**self.alibi_scaling) for i in range(n)]
            a = [1/(2**self.alibi_scaling) for i in range(n)]

            #a = [start * ratio**i for i in range(n)]
            #2.a = [elem if elem > 0.03 else 0.0625 for elem in a]

            '''
            #3.a = 0.0039
            start = 1
            ratio = (5-1) / (n-1+0.00001)
            b = [1/(2**(start + ratio*i)) for i in range(n)]
            for i in range(len(a)):
                if a[i] <= 1/(2**8):
                    if i-1 >=0 and i-1 < len(a) :
                        a[i] = b[i-1]
                    else:
                        a[i] = b[i]
                    #4. 0.0019
                    a[i] = 1/(2**9)
            '''

            #a = [start * ratio**i for i in range(n)]
            '''
            if n >= 8:
                #8t2, 8t4
                #a[7] = 1/2**2
                #a[7] = 1/2**4

                #6t2, 6t4
                #a[5] = 1/2**2
                #a[5] = 1/2**4

                #8t9, 6t9
                #a[7] = 1/2**9
                #a[5] = 1/2**9
            '''

            return a

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))

            if n == 122 :
                # original
                #[0.7071, 0.3535, 0.1768, 0.0884]
                # small
                #a = get_slopes_power_of_2(closest_power_of_2) + [0.00195, 0.003, 0.005524, 0.01104]
                #0.00195
                a = get_slopes_power_of_2(closest_power_of_2) + [0.00195, 0.3535, 0.1768, 0.0884]
                #0.00195+0.003
                #a = get_slopes_power_of_2(closest_power_of_2) + [0.00195, 0.003, 0.1768, 0.0884]
                #0.00195+0.000977
                #a = get_slopes_power_of_2(closest_power_of_2) + [0.00195, 0.000977, 0.1768, 0.0884]

                return a
            else:
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + self._get_slopes(2 * closest_power_of_2)[0::2][
                        : n - closest_power_of_2
                    ]
                )

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            a = a.to(x.device).to(x.dtype)
            slopes = self.slopes.to(a.device).to(a.dtype)
            a = a * slopes.view(self.slopes.shape[0], 1, 1)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a



###AliBi
### this is current version
#class AliBi(torch.nn.Module):
#copy from AliBi code
#class ParallelAliBi(torch.nn.Module):
class ParallelSNOPE_alibi_merge(torch.nn.Module):
    def __init__(self, neox_args):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        self.neox_args = neox_args
        self.heads = neox_args.num_attention_heads
        self.cached_matrix = None
        self.cached_seq_len = None

        self.neox_args = neox_args
        self.num_heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size

        self.alibi_scaling = neox_args.alibi_scaling
        print("self.alibi_scaling")
        print(self.alibi_scaling)

        # alibi_kernel = e^(-diff)
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ]
        print("snope alibi slopes:")
        print(slopes)
        self.register_buffer("slopes", slopes)

        #polynomial_kernel = 1/(1+a*|m-n|)^p
        self.poly_bias_a = torch.Tensor([0.01, 0.01, 0.01, 0.01, 0.0118, 0.0157, 0.0332, 0.0806, 0.2852, 0.1904, 0.3945, 0.3105])
        self.poly_bias_p = torch.Tensor([0.6914, 0.7656, 1.1406, 1.7812, 1.7266, 1.9688, 2.0938, 2.0312, 1.8672, 4.0938, 3.9375, 4.6875])

        self.bias_a = None

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        return dd
 
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))

            if n == 122 :
                #0.00195
                a = get_slopes_power_of_2(closest_power_of_2) + [0.00195, 0.3535, 0.1768, 0.0884]
                return a
            else:
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + self._get_slopes(2 * closest_power_of_2)[0::2][
                        : n - closest_power_of_2
                    ]
                )

    #log(0.5*e^(diff)+0.5*1/(1+a*|m-n|)^p)
    def alibi_merge(self, x, diff, alibi_slopes, poly_bias_a, poly_bias_p) :
        diff = diff.to(x.device).to(x.dtype)

        alibi_slopes = alibi_slopes.to(x.device).to(x.dtype)
        alibi_slopes, _ = torch.sort(alibi_slopes, 0)
        alibi_slopes = 0.5 * alibi_slopes
        alibi_slopes = alibi_slopes.view(alibi_slopes.shape[0], 1, 1)

        num_ = self.alibi_scaling
        assert num_ >= 1

        factor_ = 1.0/num_
        alibi_bias = 0
        for i in range(num_):
            c_alibi_slopes = (1/2**i)*(alibi_slopes)
            print("slopes:", i)
            print(c_alibi_slopes)

            # alibi_kernel = e^(-diff)
            a = -diff*c_alibi_slopes
            print("a.shape:")
            print(a.shape)
            alibi_bias += factor_*torch.exp(a)
            #print(alibi_bias2)

        #bias=torch.log(0.5*alibi_bias1+0.5*alibi_bias2)
        bias=torch.log(alibi_bias)

        return bias


    #log(0.5*e^(diff)+0.5*1/(1+a*|m-n|)^p)
    def alibi_kerple_merge1(self, x, diff, alibi_slopes, poly_bias_a, poly_bias_p) :
        diff = diff.to(x.device).to(x.dtype)

        alibi_slopes = alibi_slopes.to(x.device).to(x.dtype)
        alibi_slopes1, _ = torch.sort(alibi_slopes, 0)
        alibi_slopes2 = 0.5 * alibi_slopes1

        alibi_slopes1 = alibi_slopes1.view(alibi_slopes1.shape[0], 1, 1)
        alibi_slopes2 = alibi_slopes2.view(alibi_slopes2.shape[0], 1, 1)
        print("slopes1")
        print(alibi_slopes1)
        print("slopes2")
        print(alibi_slopes2)

        # alibi_kernel = e^(-diff)
        a = -diff*alibi_slopes1
        alibi_bias1 = torch.exp(a)
        #print(alibi_bias1)

        b = -diff*alibi_slopes2
        alibi_bias2 = torch.exp(b)
        #print(alibi_bias2)

        bias=torch.log(0.5*alibi_bias1+0.5*alibi_bias2)
        #bias=torch.log(alibi_bias2)

        return bias

    #log(0.5*e^(diff)+0.5*1/(1+a*|m-n|)^p)
    def alibi_kerple_merge2(self, x, diff, alibi_slopes, poly_bias_a, poly_bias_p) :
        diff = diff.to(x.device).to(x.dtype)

        alibi_slopes = alibi_slopes.to(x.device).to(x.dtype)
        alibi_slopes, _ = torch.sort(alibi_slopes, 0)
        #print(alibi_slopes)
        alibi_slopes = alibi_slopes.view(alibi_slopes.shape[0], 1, 1)

        poly_bias_a = poly_bias_a.to(x.device).to(x.dtype)
        poly_bias_a = poly_bias_a.view(poly_bias_a.shape[0], 1, 1)

        poly_bias_p = poly_bias_p.to(x.device).to(x.dtype)
        poly_bias_p = poly_bias_p.view(poly_bias_p.shape[0], 1, 1)

        # alibi_kernel = e^(-diff)
        a = -diff*alibi_slopes
        alibi_bias = torch.exp(a)
        #print("alibi_bias")
        #print(alibi_bias[11])

        #print("diff")
        #print(diff)
        # polynomial_kernel = 1/((1+a*|m-n|)^p)
        poly_bias = (1+poly_bias_a*diff).pow(-poly_bias_p)
        #print("poly_bias")
        #print(poly_bias[11])

        bias=torch.log(0.5*alibi_bias+0.5*poly_bias)
        #print("merge_bias")
        #print(torch.exp(bias[11]))

        return bias

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        '''
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )
            a = torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )
            #a = a.to(x.device).to(x.dtype)
            #slopes = self.slopes.to(a.device).to(a.dtype)
            #a = a * slopes.view(self.slopes.shape[0], 1, 1)

            a = self.alibi_kerple_merge(x, a, self.slopes, self.poly_bias_a, self.poly_bias_p)
            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]
        '''
        
        if self.cached_seq_len != seq_len_k:
            a = torch.tril(
                torch.arange(seq_len_k)
                .view(seq_len_k, 1)
                .repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1)
            )
            #a = self.alibi_kerple_merge(x, a, self.slopes, self.poly_bias_a, self.poly_bias_p)
            a = self.alibi_merge(x, a, self.slopes, self.poly_bias_a, self.poly_bias_p)

            self.cached_seq_len = seq_len_k
            self.cached_matrix = a
        else:
            a = self.cached_matrix

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a

class ParallelSNOPEKerpleLog(torch.nn.Module):
    """Kernelized T5 Relative Position Bias parallelized in the heads dimension"""

    def __init__(
        self,
        neox_args,
    ):
        super().__init__()
        self.heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size
        self.pos_emb = neox_args.pos_emb
        self.eps = 1e-2

        self.neox_args = neox_args
        print("params_dtype:%s" % (self.neox_args.params_dtype))
        self.alibi_scaling = neox_args.alibi_scaling

        # for exp_slopes
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ]

        # for gaussian kernel 
        self.gaussian_slopes = slopes
        # for gaussian end

        # for tanh_slopes
        self.tanh_slopes = torch.Tensor([0.25737, 0.12873, 0.06437, 0.03219, 0.0163, 0.00849, 0.00457, 0.00256, 0.36406, 0.18203, 0.09103, 0.04552])
        # for tanh end

        print("exp slopes:")
        print(slopes)

        print("gaussian slopes:")
        print(self.gaussian_slopes)

        print("tanh slopes:")
        print(self.tanh_slopes)

        self.register_buffer("slopes", slopes)
        
        # megatron splits across heads, so we need to make sure each head receives the correct matrix
        assert self.model_parallel_size <= self.heads and self.model_parallel_rank <= self.model_parallel_size
        
        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_p = get_parameter(2, 'uniform')
        self.bias_a = get_parameter(1, 'uniform')

        self.cached_matrix = None
        self.cached_seq_len = None

        #t5 init 
        t5_hidden_size_per_attention_head = mpu.divide(
            self.neox_args.hidden_size, self.neox_args.num_attention_heads
        )
        rpe_scale = math.sqrt(t5_hidden_size_per_attention_head)
        self.t5__init__(
            neox_args=self.neox_args,
            scale=rpe_scale,
            causal=True,
            num_buckets=self.neox_args.rpe_num_buckets,
            max_distance=self.neox_args.rpe_max_distance,
            heads=self.neox_args.num_attention_heads,
        )

    #t5 kernel
    def t5__init__(
        self,
        neox_args,
        scale,
        causal=True,
        num_buckets=32,
        max_distance=128,
        heads=8,
        init_method=init.xavier_normal_,
        ):

        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        #self.heads = heads

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        #self.t5_model_parallel_size = get_model_parallel_world_size()
        #self.t5_model_parallel_rank = get_model_parallel_rank()

        # Divide the weight matrix along the heads dimension.
        self.head_start_index, self.head_end_index = self.get_heads_range(
            self.heads, self.model_parallel_rank, self.model_parallel_size
        )
        self.t5_num_heads_per_partition = self.head_end_index - self.head_start_index

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.t5_num_heads_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_buckets,
                self.heads,
                self.t5_num_heads_per_partition,
                partition_dim=1,
                init_method=init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.t5_num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=1
            )
        self._q_len_cached = None
        self._k_len_cached = None
        self._rel_pos_bucket_cached = None

    @staticmethod
    def get_heads_range(global_n_heads, rank, world_size):
        per_partition_n_heads = divide(global_n_heads, world_size)
        index_f = rank * per_partition_n_heads
        index_l = index_f + per_partition_n_heads
        return index_f, index_l

    def _relative_position_bucket(
        self, relative_position, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not self.causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        self._rel_pos_bucket_cached = ret
        return self._rel_pos_bucket_cached
    
    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        dd.update(get_stats('bias_a', self.bias_a))
        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        dd.update(get_stats('bias_p', self.bias_p))
        return dd

    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    # t5 kernel
    def t5_kernel(self, q_len, k_len):
        if self._q_len_cached != q_len or self._k_len_cached != k_len:
            # cache bucket if first step seq len stays constant
            self._q_len_cached, self._k_len_cached = q_len, k_len
            q_pos = torch.arange(
                q_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            k_pos = torch.arange(
                k_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            rel_pos = k_pos[None, :] - q_pos[:, None]
            rp_bucket = self._relative_position_bucket(
                rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
            )
        else:
            rp_bucket = self._rel_pos_bucket_cached

        values = F.embedding(
            rp_bucket,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

        #bias = values.movedim(2, 0).unsqueeze(0)
        bias = values.movedim(2, 0)

        diff = torch.exp(bias*self.scale)
        return diff

        #return bias * self.scale

    #exp(-alpha*x)
    def exp_kernel_slopes(self, a, x):
        diff = a
        diff = diff.to(x.device)

        #print("x dtype:%s" % (x.dtype))
        #print("diff dtype:%s" % (diff.dtype))
        slopes = self.slopes.to(x.device).to(x.dtype)
        #print("exp slopes :%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff

    #exp(-0.5*alpha*x)
    def exp_kernel_slopes_half(self, a, x):
        diff = a
        diff = diff.to(x.device)

        #print("x dtype:%s" % (x.dtype))
        #print("diff dtype:%s" % (diff.dtype))
        slopes = (0.5*self.slopes).to(x.device).to(x.dtype)
        #print("exp slopes half:%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff
 
    #exp(-alpha*x^2)
    def gaussian_kernel_slopes(self, a, x):
        diff = a
        diff = diff.pow(2)
        diff = diff.to(x.device)

        #print("x dtype:%s" % (x.dtype))
        #print("diff dtype:%s" % (diff.dtype))
        #slopes = (0.5*self.gaussian_slopes).to(x.device).to(x.dtype)
        slopes = self.gaussian_slopes.to(x.device).to(x.dtype)
        #print("gaussian slopes :%s" % (slopes))
        slopes = slopes.view(self.gaussian_slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)

        return diff

    #1+tanh(-alpha*x)
    def tanh_kernel_slopes(self, a, x):
        diff = a
        diff = diff.to(x.device)

        #print("x dtype:%s" % (x.dtype))
        #print("diff dtype:%s" % (diff.dtype))
        #slopes = 0.5*self.tanh_slopes
        #slopes = slopes.to(x.device).to(x.dtype)
        slopes = self.tanh_slopes.to(x.device).to(x.dtype)
        #print("tanh slopes :%s" % (slopes))
        slopes = slopes.view(self.tanh_slopes.shape[0], 1, 1)

        epsilon = 1e-6
        CC = 0
        diff = epsilon+1+torch.tanh(-slopes*diff+CC)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff

    # The kernel has the form (1+a*|m-n|)^(-p)
    def kerplelog_kernel(self, a, x):
        diff = a

        self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        #1.
        bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        bias = torch.exp(bias)
        #2.
        #bias = (1+self.bias_a*diff).pow(-self.bias_p)
        bias = bias.to(x.device).to(x.dtype)

        return bias

    def kernel_merge(self, a, x):

        #1. exp_kernel = e^(-diff)
        exp_a = self.exp_kernel_slopes(a, x)
        #print(alibi_bias1)
        #1.1 0.5*slopes
        #exp_half_a = self.exp_kernel_slopes_half(a, x)

        #2. gaussian_kernel = e^(-diff^2)
        gaussian_a = self.gaussian_kernel_slopes(a, x)

        #3. tanh_kernel = 1+tanh(-diff+0)
        #tanh_a = self.tanh_kernel_slopes(a, x)

        #4. kerple_log kernel
        kerple_log_a = self.kerplelog_kernel(a, x)

        #5. t5 kernel
        q_len = x.shape[-2]
        k_len = x.shape[-1]
        k_len = q_len
        t5_a = self.t5_kernel(q_len, k_len)

        # merge
        #1
        #bias = torch.log(exp_half_a)
        #bias = torch.log(kerple_log_a)
        #bias = torch.log(t5_a)
        #2    
        #bias = torch.log((tanh_a+gaussian_a)/2.0)
        #bias = torch.log((exp_half_a + tanh_a)/2.0)
        #bias = torch.log((exp_half_a + gaussian_a)/2.0)
        #bias = torch.log((exp_a + gaussian_a)/2.0)

        #bias = torch.log((exp_a + kerple_log_a)/2.0)
        #bias = torch.log((gaussian_a + kerple_log_a)/2.0)
        #bias = torch.log((exp_half_a + kerple_log_a)/2.0)
        #bias = torch.log((tanh_a + kerple_log_a)/2.0)
        #bias = torch.log((t5_a + kerple_log_a)/2.0)

        #3
        #bias = torch.log((exp_half_a + tanh_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a + tanh_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a + exp_half_a + tanh_a)/3.0)
        #bias = torch.log((gaussian_a + t5_a + kerple_log_a)/3.0)

        #import
        #bias = torch.log((exp_a + exp_half_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a+(tanh_a+gaussian_a)/2.0)/2.0)

        #bias = torch.log((exp_a + gaussian_a + kerple_log_a)/3.0)
        #bias = torch.log((exp_a + exp_half_a + kerple_log_a)/3.0)
        #bias = torch.log((exp_half_a + gaussian_a + kerple_log_a)/3.0)

        #4
        #bias = torch.log((exp_a + exp_half_a + tanh_a + gaussian_a)/4.0)
        #bias = torch.log((exp_a + exp_half_a + gaussian_a + kerple_log_a)/4.0)
        #bias = (kerple_log_a)
        bias = torch.log((gaussian_a + exp_a + t5_a + kerple_log_a)/4.0)

        #5
        #bias = torch.log(((gaussian_a + exp_a + exp_half_a)/3.0 + t5_a + kerple_log_a)/3.0)

        bias = bias.to(x.device).to(x.dtype)

        return bias
   
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]
        if self.cached_seq_len != seq_len_k:
            diff = torch.tril(
                torch.arange(seq_len_k, device=x.device).view(seq_len_k, 1).repeat(1, seq_len_k)
                + torch.arange(0, -seq_len_k, -1, device=x.device)
            )
            diff = diff.to(x.dtype)
            self.cached_seq_len = seq_len_k
            self.cached_matrix = diff
        else:
            diff = self.cached_matrix
        
        #self.bias_p.data = self.bias_p.data.clamp(min=self.eps)
        #self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        #bias = -self.bias_p*torch.log(1+self.bias_a*diff) # log kernel
        
        bias = self.kernel_merge(diff, x)
        #t5 forward
        #q_len = x.shape[-2]
        #k_len = x.shape[-1]
        #k_len = q_len
        #bias = self.t5_kernel(q_len, k_len)

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            
            if type(bias) != float:
                # seq_len_k - 1 points to the last token index in the current inference batch.
                bias = bias[:, seq_len_k - 1, :].view(bias.shape[0], 1, bias.shape[2])

        return x + bias



#copy class AliBi(torch.nn.Module):
#class ParallelSNOPE(torch.nn.Module):
class ParallelSNOPE(torch.nn.Module):
    def __init__(self, neox_args):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        self.neox_args = neox_args
        print("params_dtype:%s" % (self.neox_args.params_dtype))
        self.heads = neox_args.num_attention_heads
        self.cached_matrix = None
        self.cached_seq_len = None
        self.eps = 1e-2

        self.alibi_scaling = neox_args.alibi_scaling
        self.pos_emb = neox_args.pos_emb

        self.neox_args = neox_args
        self.num_heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size

        # for exp_slopes
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ]

        # for gaussian kernel 
        #self.gaussian_slopes = torch.Tensor([0.01064, 0.00269, 0.00068, 0.00017, 6e-05, 3e-05, 2e-05, 1e-05, 0.02143, 0.00536, 0.00135, 0.00034]) 
        self.gaussian_slopes = slopes
        # for gaussian end

        # for tanh_slopes
        self.tanh_slopes = torch.Tensor([0.25737, 0.12873, 0.06437, 0.03219, 0.0163, 0.00849, 0.00457, 0.00256, 0.36406, 0.18203, 0.09103, 0.04552])
        #self.tanh_slopes = slopes
        # for tanh end

        print("exp slopes:")
        print(slopes)

        print("gaussian slopes:")
        print(self.gaussian_slopes)

        print("tanh slopes:")
        print(self.tanh_slopes)

        self.register_buffer("slopes", slopes)
        self.bias_a = None

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        #self.bias_a = get_parameter(1, 'uniform')

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        return dd
 
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    #exp(-alpha*x)
    def exp_kernel_slopes(self, a, x):
        diff = a
        diff = diff.to(x.device)

        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        slopes = self.slopes.to(x.device).to(x.dtype)
        print("exp slopes :%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff

    #exp(-0.5*alpha*x)
    def exp_kernel_slopes_half(self, a, x):
        diff = a
        diff = diff.to(x.device)

        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        slopes = (0.5*self.slopes).to(x.device).to(x.dtype)
        print("exp slopes half:%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff
 
    #exp(-alpha*x^2)
    def gaussian_kernel_slopes(self, a, x):
        diff = a
        diff = diff.pow(2)
        diff = diff.to(x.device)

        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        #slopes = (0.5*self.gaussian_slopes).to(x.device).to(x.dtype)
        slopes = self.gaussian_slopes.to(x.device).to(x.dtype)
        print("gaussian slopes :%s" % (slopes))
        slopes = slopes.view(self.gaussian_slopes.shape[0], 1, 1)

        diff = torch.exp(-slopes*diff)
        diff = diff.to(x.device).to(x.dtype)

        return diff

    #1+tanh(-alpha*x)
    def tanh_kernel_slopes(self, a, x):
        diff = a
        diff = diff.to(x.device)

        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        #slopes = 0.5*self.tanh_slopes
        #slopes = slopes.to(x.device).to(x.dtype)
        slopes = self.tanh_slopes.to(x.device).to(x.dtype)
        print("tanh slopes :%s" % (slopes))
        slopes = slopes.view(self.tanh_slopes.shape[0], 1, 1)

        epsilon = 1e-6
        CC = 0
        diff = epsilon+1+torch.tanh(-slopes*diff+CC)
        diff = diff.to(x.device).to(x.dtype)
        #print(diff)

        return diff

    def kernel_merge(self, a, x):

        #1. exp_kernel = e^(-diff)
        exp_a = self.exp_kernel_slopes(a, x)
        #print(alibi_bias1)
        #1.1 0.5*slopes
        exp_half_a = self.exp_kernel_slopes_half(a, x)

        #2. gaussian_kernel = e^(-diff^2)
        gaussian_a = self.gaussian_kernel_slopes(a, x)

        #3. tanh_kernel = 1+tanh(-diff+0)
        #tanh_a = self.tanh_kernel_slopes(a, x)

        # merge
        #1
        #bias = torch.log(exp_half_a)
        #2    
        #bias = torch.log((tanh_a+gaussian_a)/2.0)
        #bias = torch.log((exp_half_a + tanh_a)/2.0)
        #bias = torch.log((exp_half_a + gaussian_a)/2.0)
        #bias = torch.log((exp_a + gaussian_a)/2.0)
        #3
        #bias = torch.log((exp_half_a + tanh_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a + tanh_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a + exp_half_a + tanh_a)/3.0)

        bias = torch.log((exp_a + exp_half_a + gaussian_a)/3.0)
        #bias = torch.log((exp_a+(tanh_a+gaussian_a)/2.0)/2.0)
        #4
        #bias = torch.log((exp_a + exp_half_a + tanh_a + gaussian_a)/4.0)
        bias = bias.to(x.device).to(x.dtype)

        return bias

    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )

            a = torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )

            #use this: a = self.tanh_kernel(-a)
            a = self.kernel_merge(a, x)
            #a = self.tanh_kernel_slopes(a, x)
            #a = self.gaussian_kernel_slopes(a, x)

            ### for alibi
            #a = a.to(x.device).to(x.dtype)
            #slopes = self.slopes.to(a.device).to(a.dtype)
            #a = -a * slopes.view(self.slopes.shape[0], 1, 1)
            ### for alibi

            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a

#copy class AliBi(torch.nn.Module):
#2024.02.09
class ParallelSNOPE_right(torch.nn.Module):
    def __init__(self, neox_args):
        super().__init__()
        # megatron splits across heads, so we need to make sure each
        # head receives the correct matrix
        self.neox_args = neox_args
        print("params_dtype:%s" % (self.neox_args.params_dtype))
        self.heads = neox_args.num_attention_heads
        self.cached_matrix = None
        self.cached_seq_len = None
        self.eps = 1e-2

        self.alibi_scaling = neox_args.alibi_scaling
        self.pos_emb = neox_args.pos_emb

        self.neox_args = neox_args
        self.num_heads = neox_args.num_attention_heads
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()
        self.num_heads_per_partition = self.heads // self.model_parallel_size

        # for tanh_slopes
        slopes = torch.Tensor(self._get_slopes(self.heads))[
            self.model_parallel_rank * self.num_heads_per_partition : (self.model_parallel_rank + 1) * self.num_heads_per_partition
        ]
        # for tanh_slopes
        # for gaussian kernel 
        #slopes = torch.Tensor([0.010638297872340425, 0.002688172043010753, 0.0006756756756756757, 0.00016937669376693767, 6.103515625e-05, 3.0517578125e-05, 1.52587890625e-05, 7.62939453125e-06, 0.02142747821777417, 0.005356869554443543, 0.0013494404221117322, 0.0003386526729820632])
        #slopes = torch.Tensor([0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425, 0.010638297872340425])
        slopes = torch.Tensor([0.01064, 0.00269, 0.00068, 0.00017, 6e-05, 3e-05, 2e-05, 1e-05, 0.02143, 0.00536, 0.00135, 0.00034]) 
        # for gaussian end

        # for tanh
        #slopes = torch.Tensor([0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178, 0.2573739082269178])
        #slopes = torch.Tensor([0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087, 0.12872659589188087]
        #slopes = torch.Tensor([0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465, 0.0025616027933590465])
        #slopes = torch.Tensor([0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141, 0.004574167158698141])
        #slopes = torch.Tensor([0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256])
        #slopes = torch.Tensor([0.25737, 0.12873, 0.06437, 0.03219, 0.0163, 0.00849, 0.00457, 0.00256, 0.36406, 0.18203, 0.09103, 0.04552])
        #slopes = torch.Tensor([0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003])
        #slopes = torch.Tensor([0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256, 0.00256])
        # for tanh end

        print("slopes:")
        print(slopes)
        self.register_buffer("slopes", slopes)

        self.bias_a = None

        # Allocate weights and initialize.
        # The kernel has the form -p*log(1+a*|m-n|)
        def get_parameter(scale, init_method):
            if init_method == 'ones':
                return Parameter(torch.ones(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
            elif init_method == 'uniform':
                return Parameter(torch.rand(
                               self.num_heads_per_partition,
                               device=torch.cuda.current_device(),
                               dtype=neox_args.params_dtype,
                               )[:,None,None]*scale )
        
        self.bias_a = get_parameter(1, 'uniform')

    def stats(self):
        def get_stats(name, obj):
            return {name+'_mean': obj.mean().detach().cpu(),
                    name+'_std': obj.std().detach().cpu(),
                    name+'_max': obj.max().detach().cpu(),
                    name+'_min': obj.min().detach().cpu()}
        dd = {}
        if self.bias_a is not None:
            self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
            dd.update(get_stats('bias_a', self.bias_a))

        return dd
 
    def _get_slopes(self, n):
        """
        Get slopes for Alibi positional embedding
        n : int = number of heads.
        For best performance, restrict n to a power of 2.
        """

        def get_slopes_power_of_2(n):
            #start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            #ratio = start
            #return [start * ratio**i for i in range(n)]
            epsilon = 1e-6
            start = 1
            ratio = (3-start) / (n-1+epsilon)
            #return [start + ratio*i for i in range(n)]

            return [1 for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + self._get_slopes(2 * closest_power_of_2)[0::2][
                    : n - closest_power_of_2
                ]
            )

    def _get_slopes2(self, n):
        def get_slopes_power_of_2(n):
            epsilon = 1e-6
            start = 2.5
            ratio = (3-start) / (n-1+epsilon)
            return [start + ratio*i for i in range(n)]

        return get_slopes_power_of_2(n)
 
    #Matrix a's value must be negative numbers
    def to_gaussian_distribution(self, a, seq_len_k) :
        cseq = torch.arange(seq_len_k)
        mean_value = torch.mean(cseq.to(torch.float32))
        std_value = torch.std(cseq.to(torch.float32))
        epsilon = 1e-6

        #position 1's gaussian value equals to 1.7270
        zero = -((0-mean_value) / (std_value + epsilon))
        diff = torch.tril((a-mean_value) / (std_value + epsilon)+zero)

        return diff

    #Matrix a's value must be positive numbers
    def to_min_max_distribution(self, a, seq_len_k) :
        min_value = 0
        max_value = seq_len_k-1
        epsilon = 1e-6

        diff = torch.tril((a-min_value) / (max_value-min_value))
        return diff

    #f(x) = 1/((1+ax)^p)
    def polynomial_kernel(self, diff):
        a = 0.1
        p = 2.2
        return -p*torch.log(1+a*diff)

    def linear_kernel():
        return

    #exp(-alpha*x^2)
    def gaussian_kernel_slopes(self, a, x):

        diff = -a
        diff = diff.pow(2)
        diff = diff.to(x.device)

        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        slopes = self.slopes.to(x.device).to(x.dtype)
        print("slopes :%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = -slopes*diff
        diff = diff.to(x.device).to(x.dtype)

        return diff

    def tanh_kernel_learning(self, a):
        a = -a
        seq_len_k = 512
        epsilon = 1e-6

        diff = self.to_gaussian_distribution(a, seq_len_k)

        CC = 0
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)
        diff = torch.tril(torch.log(epsilon+1+torch.tanh(self.bias_a*diff+CC)))

        return diff

    def tanh_kernel_slopes2(self, a, x):
        a = -a
        seq_len_k = 512
        epsilon = 1e-6

        diff = self.to_gaussian_distribution(a, seq_len_k)

        CC = 0
        gamma = 2.5

        diff = diff.to(x.device)
        slopes = self.slopes.to(x.device).to(x.dtype)
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.log(epsilon+1+torch.tanh(slopes*diff+CC))
        diff = diff.to(x.device).to(x.dtype)

        return diff

    def tanh_kernel_slopes(self, a, x):
        epsilon = 1e-6
        CC = 0

        diff = -a
        diff = diff.to(x.device)
        print("x dtype:%s" % (x.dtype))
        print("diff dtype:%s" % (diff.dtype))
        slopes = self.slopes.to(x.device).to(x.dtype)
        print("slopes :%s" % (slopes))
        slopes = slopes.view(self.slopes.shape[0], 1, 1)

        diff = torch.log(epsilon+1+torch.tanh(slopes*diff+CC))
        diff = diff.to(x.device).to(x.dtype)

        return diff

    # The closer the distance(distance=0), the higher the similarity(similarity=1).
    # For example, if the distance is equal to 0 and the similarity is equal to 1. Distance equals 511, similarity equals 0
    # matrix a's value must be negative numbers
    #
    # x=-(i-j)
    # x=to_gaussian_distribution(x), -1<x<0
    # f(x)=1+tanh(gamma*x+cc), -1<x<0, 0<f(x)<1
    # eg. tanh_kernel(x=0)=1, tanh_kernel(x=511)=0
    #
    def tanh_kernel2(self, a):
        a = -a
        seq_len_k = 512
        diff = self.to_gaussian_distribution(a, seq_len_k)

        epsilon = 1e-6
        gamma = 1.5
        CC = 0

        gamma = 6
        CC = 0
        gamma = 3
        CC = 0
        gamma = 7
        CC = 0
        gamma = 2.5
        CC = 0
        diff = torch.log(epsilon+1+torch.tanh(gamma*diff+CC))

        return diff 

    def tanh_kernel(self, a):
        a = -a
        gamma = -0.25
        CC = 0
        diff = torch.log(epsilon+1+torch.tanh(gamma*diff+CC))

        return diff 

    # f(x) = 1 + tanh(-0.01x)
    def tanh_kernel_log(self, a):
        epsilon = 1e-6
        gamma = -0.01
        C = 0

        return torch.log(epsilon+1+torch.tanh(gamma*a+C))

    # The closer the distance(distance=0), the higher the similarity(similarity=1).
    # For example, if the distance is equal to 0 and the similarity is equal to. Distance equals 511, similarity equals 0
    # matrix a's value must be positive numbers
    #
    # x=(i-j)
    # x=to_min_max_distribution(x), 0<x<1
    # f(x)=1-tanh(gamma*x+cc), 0<x<1, 1>f(x)>0
    # eg. tanh_kernel(x=0)=1, tanh_kernel(x=511)=0
    def tanh_kernel_minmax(self, a):
        seq_len_k = 512
        diff = self.to_min_max_distribution(a, seq_len_k)

        epsilon = 1e-6
        gamma = 8
        CC = 0

        diff = torch.tril(torch.log(epsilon+1-torch.tanh(gamma*diff+CC)))

        return diff 

    def tanh_kernel_minmax_learning(self, a, x):
        seq_len_k = 512
        diff = self.to_min_max_distribution(a, seq_len_k)

        epsilon = 1e-6
        gamma = 8
        CC = 0

        diff = diff.to(x.device).to(x.dtype)
        self.bias_a.data = self.bias_a.data.clamp(min=self.eps)

        diff = torch.log(epsilon+1-torch.tanh(bias_a*diff+CC))
 
    def forward(self, x):
        # [b, np, sq, sk]
        seq_len_q = x.shape[-2]
        seq_len_k = x.shape[-1]

        # Initialize the AliBi matrix to match the first provided key length; grow it exponentially
        # afterwards if longer inputs are provided. This is important for inference, where we will
        # encounter progressively longer samples; it should have no effect at training time.
        if self.cached_seq_len is not None and self.cached_seq_len >= seq_len_k:
            a = self.cached_matrix
        else:
            target_seq_len = (
                seq_len_k if self.cached_seq_len is None else self.cached_seq_len * 4
            )

            a = -torch.tril(
                torch.arange(target_seq_len)
                .view(target_seq_len, 1)
                .repeat(1, target_seq_len)
                + torch.arange(0, -target_seq_len, -1)
            )

            if 'norm' in self.pos_emb:
                '''
                diff = -a
                print('snope norm, var')
                # for test
                #my_seq_len_k = seq_len_k
                #if my_seq_len_k > 512:
                #    my_seq_len_k = 512
                my_seq_len_k = 512

                cseq = torch.arange(my_seq_len_k)
                s_mean = cseq.to(torch.float32).mean()
                s_variance = cseq.to(torch.float32).pow(2).mean()
                #s_mean = torch.mean(cseq.to(torch.float32))
                #s_variance = torch.var(cseq.to(torch.float32))
                #s_mean = torch.Tensor([255.5])
                #s_variance = torch.Tensor([21888.0])
                epsilon = 1e-6

                a = -torch.tril((diff-s_mean) * torch.rsqrt(s_variance + epsilon))
                '''
                print('norm in in in')
                #use this: a = self.tanh_kernel(-a)
                #a = self.gaussian_kernel_slopes(-a, x)
                #a = self.tanh_kernel(-a, x)
                #a = self.tanh_kernel_slopes(-a, x)
                #a = self.polynomial_kernel(-a)

            ### for alibi
            #a = a.to(x.device).to(x.dtype)
            #slopes = self.slopes.to(a.device).to(a.dtype)
            #a = a * slopes.view(self.slopes.shape[0], 1, 1)
            ### for alibi

            self.cached_seq_len = target_seq_len
            self.cached_matrix = a

        # If the AliBi matrix is larger than the key length, clip it.
        if self.cached_seq_len > seq_len_k:
            a = self.cached_matrix[:, :seq_len_k, :seq_len_k]

        if seq_len_q != seq_len_k:
            # In the train case x has dimensionality [b, np, sq, sk] with sq == sk
            # The number of query tokens is equal to the number of key tokens
            # At inference time with cache in layer_past sq is not equal to sk. sq only contains one token (the last one in the full sequence)
            # In this case we use the appropriate token index of the cache matrix.
            # As the cache matrix could already be bigger from a past inference, not the last token index in the sq sequence is used
            assert (
                seq_len_q == 1
            ), "assumption sq == sk unless at inference time with cache in layer_past with sq == 1"
            a = a[:, seq_len_k - 1, :].view(
                a.shape[0], 1, a.shape[2]
            )  # seq_len_k - 1 points to the last token index in the current inference batch.

        return x + a

class ParallelRelativePositionBias(torch.nn.Module):
    """T5 Relative Position Bias parallelized in the heads dimension

    Based on https://github.com/lucidrains/x-transformers/blob/6b93c21be0d0a679da6f7b9621d9bb638ab18428/x_transformers/x_transformers.py#L106 (14.12.2021)
    and adapted for megatron's model parallelism

    Arguments:
        scale: scaling factor for the bias
        causal: flag for causal/non-causal language modelling.
        num_buckets: number of rp buckets.
        max_distance: max distance in sequence dim for each bucket.
        heads: number of attention heads (total)
    """

    def __init__(
        self,
        neox_args,
        scale,
        causal=True,
        num_buckets=32,
        max_distance=128,
        heads=8,
        init_method=init.xavier_normal_,
    ):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.heads = heads

        # Set the defaults for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        self.model_parallel_rank = get_model_parallel_rank()

        # Divide the weight matrix along the heads dimension.
        self.head_start_index, self.head_end_index = self.get_heads_range(
            self.heads, self.model_parallel_rank, self.model_parallel_size
        )
        self.num_heads_per_partition = self.head_end_index - self.head_start_index

        # Allocate weights and initialize.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.num_buckets,
                self.heads,
                self.num_heads_per_partition,
                partition_dim=1,
                init_method=init_method,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_buckets,
                    self.num_heads_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=1
            )
        self._q_len_cached = None
        self._k_len_cached = None
        self._rel_pos_bucket_cached = None

    @staticmethod
    def get_heads_range(global_n_heads, rank, world_size):
        per_partition_n_heads = divide(global_n_heads, world_size)
        index_f = rank * per_partition_n_heads
        index_l = index_f + per_partition_n_heads
        return index_f, index_l

    def _relative_position_bucket(
        self, relative_position, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        if not self.causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, n, val_if_large)
        self._rel_pos_bucket_cached = ret
        return self._rel_pos_bucket_cached

    def forward(self, q_len, k_len):
        if self._q_len_cached != q_len or self._k_len_cached != k_len:
            # cache bucket if first step seq len stays constant
            self._q_len_cached, self._k_len_cached = q_len, k_len
            q_pos = torch.arange(
                q_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            k_pos = torch.arange(
                k_len, dtype=torch.long, device=torch.cuda.current_device()
            )
            rel_pos = k_pos[None, :] - q_pos[:, None]
            rp_bucket = self._relative_position_bucket(
                rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
            )
        else:
            rp_bucket = self._rel_pos_bucket_cached
        values = F.embedding(
            rp_bucket,
            self.weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        bias = values.movedim(2, 0).unsqueeze(0)
        return bias * self.scale


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        gather_output=True,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
    ):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.output_size_per_partition,
                0,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=stride
            )

        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=neox_args.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def set_parallel_output(self, value: bool):
        assert isinstance(value, bool)
        self.gather_output = (
            not value
        )  # if gather_output is True, parallel output is False, so we set the opposite

    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        neox_args,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        parallel_output=False,
    ):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.parallel_output = parallel_output

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        if neox_args.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=neox_args.params_dtype,
                )
            )
            self.master_weight = _initialize_affine_weight_cpu(
                neox_args,
                self.weight,
                self.output_size,
                self.input_size,
                self.input_size_per_partition,
                1,
                init_method,
                stride=stride,
                return_master_weight=keep_master_weight_for_test,
            )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=neox_args.params_dtype,
                )
            )
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=1, stride=stride
            )
        if bias:
            if neox_args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, dtype=neox_args.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
                )
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def set_parallel_output(self, parallel_output: bool):
        assert isinstance(parallel_output, bool)
        self.parallel_output = parallel_output

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        if not self.parallel_output:
            output_ = reduce_from_model_parallel_region(output_parallel)
        else:
            output_ = output_parallel
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
