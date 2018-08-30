
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function, Variable, gradcheck


class adaptive(Function):

	@staticmethod
	def forward(ctx, in_prob, out_prob, eps=0.01):
		# in_prob: sum of previous halting scores (h)
		# out_prob: current halting scores
		total_prob = in_prob + out_prob
		mask = total_prob >= (1 - eps)
		updated_out_prob = out_prob.clone().masked_scatter_(mask, 1 - in_prob.masked_select(mask))

		ctx.supported_variables = mask

		return updated_out_prob

	@staticmethod
	def backward(ctx, grad_updated_out_prob):
		mask = ctx.supported_variables
		inv_mask = mask.eq(0)
		grad_in_prob = grad_out_prob = None

		if ctx.needs_input_grad[0]:
			grad_in_prob = torch.zeros_like(grad_updated_out_prob)
			grad_in_prob.data.masked_scatter_(mask, -grad_updated_out_prob.data.masked_select(mask))
		if ctx.needs_input_grad[1]:
			grad_out_prob = torch.zeros_like(grad_updated_out_prob)
			grad_out_prob.data.masked_scatter_(inv_mask, grad_updated_out_prob.data.masked_select(inv_mask))

		return grad_in_prob, grad_out_prob


class AdaptiveNet(nn.Module):

	def __init__(self, dim):
		super(AdaptiveNet, self).__init__()
		# Generate halting score in each step (A MLP of 2 layers is used here as the halting predictor)
		self.halting_predict = nn.Sequential(
				nn.Linear(dim, dim),
				nn.ReLU(inplace=True),
				nn.Linear(dim, 1),
				nn.Sigmoid(),
			)

	def forward(self, cum_prob, data):
		# data: the generated tensor at each time step of RNN or Transformer
		# cum_prob: the sum of halting scores of all previous time steps

		return adaptive.apply(cum_prob, self.halting_predict(data))


if __name__ == "__main__":
	batch = 1000
	num_seq = 10
	torch.manual_seed(8123)
	cum_prob = Variable(torch.zeros(batch).double(), requires_grad=True)
	steps = [Variable(F.sigmoid(torch.randn(batch).double() * 10), requires_grad=True) for _ in range(num_seq)]

	for i in range(num_seq):
		input = (cum_prob, steps[i])
		out_prob = adaptive.apply(cum_prob, steps[i])
		test = gradcheck(adaptive.apply, input)
		cum_prob = (cum_prob + out_prob)
		cum_prob = Variable(cum_prob.data, requires_grad=True)
		print("test:", test)
