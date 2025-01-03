# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" FM
Reference:
	'Factorization Machines', Steffen Rendle, 2010 IEEE International conference on data mining.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextCTRModel, ContextModel

class FMBase(object):
	@staticmethod
	def parse_model_args_FM(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		return parser

	def _define_init_params(self, args,corpus):
		self.vec_size = args.emb_size
		self._define_params_FM()
		self.apply(self.init_weights)
	
	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self._define_params_FM()
		self.apply(self.init_weights)
	
	def _define_params_FM(self):	
		self.context_embedding = nn.ModuleDict()
		self.linear_embedding = nn.ModuleDict()
		for f in self.context_features:
			self.context_embedding[f] = nn.Embedding(self.feature_max[f],self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,self.vec_size,bias=False)
			self.linear_embedding[f] = nn.Embedding(self.feature_max[f],1) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1,1,bias=False)
		self.overall_bias = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)

	def _get_embeddings_FM(self, feed_dict):
		item_ids = feed_dict['item_id']
		_, item_num = item_ids.shape

		fm_vectors = [self.context_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id') 
						  else self.context_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		fm_vectors = torch.stack([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1) 
							for v in fm_vectors], dim=-2) # batch size * item num * feature num * feature dim: 84,100,2,64
		linear_value = [self.linear_embedding[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
							else self.linear_embedding[f](feed_dict[f].float().unsqueeze(-1)) for f in self.context_features]
		linear_value = torch.cat([v if len(v.shape)==3 else v.unsqueeze(dim=-2).repeat(1, item_num, 1)
	  				for v in linear_value],dim=-1) # batch size * item num * feature num
		linear_value = self.overall_bias + linear_value.sum(dim=-1)
		return fm_vectors, linear_value

	def forward(self, feed_dict):
		fm_vectors, linear_value = self._get_embeddings_FM(feed_dict)
		fm_vectors = 0.5 * (fm_vectors.sum(dim=-2).pow(2) - fm_vectors.pow(2).sum(dim=-2))
		predictions = linear_value + fm_vectors.sum(dim=-1)
		return {'prediction':predictions}

class FM_ADTCTR(ContextCTRModel, FMBase):
	reader, runner = 'ContextReader', 'ADTRunner'
	extra_log_args = ['emb_size','loss_n']

	@staticmethod
	def parse_model_args(parser):
		parser = FMBase.parse_model_args_FM(parser)
		parser.add_argument('--paradigm', type=str, default='CE',
							help='Two paradigms to formulate loss functions {T_CE, R_CE}')
		parser.add_argument('--drop_rate', type=float, default=0.1,
							help='drop rate')
		parser.add_argument('--exponent', type = float, default = 1, 
							help='exponent of the drop rate {0.5, 1, 2}')
		parser.add_argument('--beta', type=float, default=0.1,
							help='beta')
		parser.add_argument('--num_gradual', type = int, default = 30000,
							help='how many epochs to linearly increase drop_rate')
		return ContextCTRModel.parse_model_args(parser)

	def __init__(self, args, corpus):
		ContextCTRModel.__init__(self, args, corpus)
		self.count = 0
		self.paradigm = args.paradigm
		self.drop_rate = args.drop_rate
		self.exponent = args.exponent
		self.beta = args.beta
		self.num_gradual = args.num_gradual
		self.loss_fn = nn.BCELoss()
		self._define_init(args,corpus)

	# define drop rate schedule
	def drop_rate_schedule(self, iteration):
		drop_rate = np.linspace(0, self.drop_rate**self.exponent, self.num_gradual)
		if iteration < self.num_gradual:
			return drop_rate[iteration]
		else:
			return self.drop_rate

	def loss(self, out_dict: dict) -> torch.Tensor:
		if self.paradigm == 'T_CE':
			y = out_dict['prediction']
			t = out_dict['label'].float()
			loss = F.binary_cross_entropy(y, t, reduce=False)
			loss_mul = loss * t
			ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
			loss_sorted = loss[ind_sorted]
			remember_rate = 1 - self.drop_rate_schedule(self.count)
			num_remember = int(remember_rate * len(loss_sorted))
			ind_update = ind_sorted[:num_remember]
			loss = self.loss_fn(y[ind_update], t[ind_update])

		elif self.paradigm == 'R_CE':
			y = out_dict['prediction']
			t = out_dict['label'].float()
			loss = F.binary_cross_entropy(y, t, reduce=False)
			y_hat = y.detach()
			weight = torch.pow(y_hat, self.beta) * t + torch.pow((1 - y_hat), self.beta) * (1 - t)
			loss = torch.mean(loss * weight)

		elif self.paradigm == 'CE':
			loss = self.loss_fn(out_dict['prediction'], out_dict['label'].float())

		else:
			raise ValueError('Invalid paradigm: {}'.format(self.paradigm))
		
		return loss

	def forward(self, feed_dict):
		out_dict = FMBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict
	