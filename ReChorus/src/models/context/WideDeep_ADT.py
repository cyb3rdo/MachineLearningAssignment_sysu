# -*- coding: UTF-8 -*-
# @Author : Jiayu Li 
# @Email  : jy-li20@mails.tsinghua.edu.cn

""" WideDeep
Reference:
  Wide {\&} Deep Learning for Recommender Systems, Cheng et al. 2016. The 1st workshop on deep learning for recommender systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from models.BaseContextModel import ContextModel, ContextCTRModel
from models.context.FM import FMBase
from utils.layers import MLP_Block

class WideDeepBase(FMBase):
	@staticmethod
	def parse_model_args_WD(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors.')
		parser.add_argument('--layers', type=str, default='[64]',
							help="Size of each layer.")
		return parser

	def _define_init(self, args, corpus):
		self._define_init_params(args,corpus)
		self.layers = eval(args.layers)
		self._define_params_WD()
		self.apply(self.init_weights)

	def _define_params_WD(self):
		self._define_params_FM()
		pre_size = len(self.context_features) * self.vec_size
		# deep layers
		self.deep_layers = MLP_Block(pre_size, self.layers, hidden_activations="ReLU",
							   batch_norm=False, dropout_rates=self.dropout, output_dim=1)
	
	def forward(self, feed_dict):
		deep_vectors, wide_prediction = self._get_embeddings_FM(feed_dict)
		deep_vector = deep_vectors.flatten(start_dim=-2)
		deep_prediction = self.deep_layers(deep_vector).squeeze(dim=-1)
		predictions = deep_prediction + wide_prediction
		return {'prediction':predictions}
		
class WideDeep_ADTCTR(ContextCTRModel, WideDeepBase):
	reader, runner = 'ContextReader', 'ADTRunner'
	extra_log_args = ['emb_size','layers','loss_n']
	@staticmethod
	def parse_model_args(parser):
		parser = WideDeepBase.parse_model_args_WD(parser)
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
		return ContextModel.parse_model_args(parser)
    
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
		out_dict = WideDeepBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict
