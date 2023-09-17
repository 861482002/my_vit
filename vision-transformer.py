# -*- codeing = utf-8 -*-
# @Time : 2023-09-17 14:28
# @Author : 张庭恺
# @File : vision-transformer.py
# @Software : PyCharm
import copy
import thop
import torch
import torch.nn as nn
import torchvision
from typing import Callable, List, Optional


class MLP(nn.Module):
	def __init__(self, in_c, up_scale=4, drop_out=0.2):
		super(MLP, self).__init__()
		self.mlp = nn.Sequential(nn.Linear(in_c, out_features=(in_c * up_scale)),
		                         nn.GELU(),
		                         nn.Dropout(drop_out),
		                         nn.Linear((in_c * up_scale), in_c),
		                         nn.Dropout(drop_out),
		                         )

	def forward(self, x):
		x = self.mlp(x)
		return x


class VIT_encoder_layer(nn.Module):
	# in_c,num_head,up_scale,drop_out
	def __init__(self, in_c, num_head, up_scale=4, drop_out=0.1):
		super(VIT_encoder_layer, self).__init__()
		self.layer_norm1 = nn.LayerNorm(in_c)
		self.msa = MSA(in_c, num_head)
		self.dp1 = nn.Dropout(drop_out)

		self.layer_norm2 = nn.LayerNorm(in_c)
		self.mlp = MLP(in_c, up_scale, drop_out)
		self.dp2 = nn.Dropout(drop_out)

	def forward(self, src):
		x1 = self.layer_norm1(src)
		x_atten, x_weight = self.msa(x1)
		x1 = self.dp1(x_atten)

		src = src + x1

		x2 = self.layer_norm2(src)
		x2 = self.dp2(self.mlp(x2))

		return src + x2

		pass


def clone(model, num_encoder):
	return nn.ModuleList([copy.deepcopy(model) for _ in range(num_encoder)])


def self_atten(q, k, v):
	# shape = [b , num_head , sqe_len , sub_dim]
	b, num_head, seq_len, sub_dim = q.shape

	# shape = [b,num_head,sqe_len,sub_dim]
	atten_weight = torch.matmul(q, torch.transpose(k, -1, -2)) / torch.sqrt(torch.tensor(sub_dim))

	# shape = [b, num_head , sqe_len , sqe_len]
	atten_weight = torch.softmax(atten_weight, -1)

	# shqpe = [b , num_head , sqe_len , sub_dim]
	atten_src = torch.matmul(atten_weight, v)

	return atten_src, atten_weight
	pass


class MSA(nn.Module):
	def __init__(self, in_c, num_head):
		super(MSA, self).__init__()
		assert in_c % num_head == 0
		self.num_head = num_head
		self.sub_dim = in_c / num_head
		self.Q = nn.Linear(in_c, in_c)
		self.K = nn.Linear(in_c, in_c)
		self.V = nn.Linear(in_c, in_c)
		self.sa = self_atten

	def transpose_head_1(self, tensor: torch.Tensor, num_head):
		b, sqe_len, dim = tensor.shape
		return tensor.view(b, sqe_len, num_head, -1).transpose(1, 2)

	def transpose_head_2(self, tensor: torch.Tensor):
		b, num_head, sqe_len, sub_dim = tensor.shape
		return tensor.transpose(1, 2).contiguous().view(b, sqe_len, -1)

	def forward(self, src):
		batch_size, length, dim = src.shape
		q, k, v = self.Q(src), self.K(src), self.V(src)
		q = self.transpose_head_1(q, self.num_head)
		k = self.transpose_head_1(k, self.num_head)
		v = self.transpose_head_1(v, self.num_head)

		# q = q.view(batch_size,length,self.num_head, self.sub_dim).transpose(1,2)
		# k = k.view(batch_size,length,self.num_head, self.sub_dim).transpose(1,2)
		# v = v.view(batch_size,length,self.num_head, self.sub_dim).transpose(1,2)
		atten_src, atten_weight = self.sa(q, k, v)

		atten_src = self.transpose_head_2(atten_src)
		atten_weight = self.transpose_head_2(atten_weight)

		return atten_src, atten_weight

		pass


class VIT_encoder(nn.Module):
	# in_c,num_encoder,num_head,up_scale,drop_out
	def __init__(self, in_c, num_encoder, num_head, up_scale, drop_out):
		super(VIT_encoder, self).__init__()

		self.encoder_layer = VIT_encoder_layer(in_c, num_head, up_scale, drop_out)

		self.vit_encoders = clone(self.encoder_layer, num_encoder)

		self.vit_encoders_sequential = nn.Sequential(*self.vit_encoders,
		                                             )

		# def sequential(self,model_list : Callable[...,nn.Module]):
		# 	return nn.Sequential(model_list)

		pass

	def forward(self, x):
		x = self.vit_encoders_sequential(x)
		return x
		pass


class VIT(nn.Module):
	def __init__(self, in_c, patch_size, num_cls, num_head, up_scale, num_encoder, drop_out=0.2):
		super(VIT, self).__init__()
		self.in_c = in_c
		self.out_c = in_c * patch_size ** 2
		self.conv = nn.Conv2d(self.in_c, self.out_c, kernel_size=patch_size, stride=patch_size)
		self.class_token = nn.Parameter(torch.zeros(1, 1, self.out_c), requires_grad=True)

		self.position_embedding = nn.Parameter(torch.zeros(1, 1, self.out_c), requires_grad=True)
		self.dp1 = nn.Dropout(drop_out)

		self.tr_encoder = VIT_encoder(self.out_c, num_encoder, num_head, up_scale, drop_out)
		self.layer_norm = nn.LayerNorm(self.out_c)

		self.cls_head = nn.Linear(self.out_c, num_cls)

		nn.init.trunc_normal_(self.class_token, std=0.02)
		nn.init.trunc_normal_(self.position_embedding, std=0.02)

	def forward(self, x):
		# x  shape = [b , 3 , 224 , 224]

		# shape = [b , 768 , 14,14]
		x = self.conv(x)
		# shape = [b , 196 , 768]
		x = x.flatten(2).transpose(-1, -2)

		# cla_embedding
		class_token = self.class_token.repeat(x.size(0), 1, 1)
		x_embedding = self.dp1(torch.cat([x, class_token], dim=1) + self.position_embedding)

		x = self.layer_norm(self.tr_encoder(x_embedding))

		x_logic = x[:,0]
		cls_prob = torch.softmax(self.cls_head(x_logic),-1)

		return cls_prob
		pass


if __name__ == '__main__':
	x = torch.ones(1,3, 224, 224)
	model = VIT(3 , 16 , 10 , 8 , 4 , 6 , 0.2)

	mac , param = thop.profile(model,(x,))
	print(mac , param)
	# for name , p in model.named_parameters():
	# 	if name.startswith('tr'):
	# 		p.requires_grad = False
	# 		print(p.requires_grad)

	print(model.tr_encoder.vit_encoders_sequential[0])
	# y = model(x)
	# print(y.shape)
