import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from torch.autograd import Variable

import util
from training_data_wrapper import *
from drugcell_nn import *
from ccc_loss import *


class VNNTrainer():

	def __init__(self, data_wrapper):
		self.data_wrapper = data_wrapper
		self.train_feature = self.data_wrapper.train_feature
		self.train_label = self.data_wrapper.train_label
		self.val_feature = self.data_wrapper.val_feature
		self.val_label = self.data_wrapper.val_label
		self.task = self.data_wrapper.task


	def _get_loss_fn(self):
		if self.task == 'binary':
			return nn.BCEWithLogitsLoss()
		else:
			return nn.MSELoss()


	def _get_aux_loss_fn(self):
		"""Loss for auxiliary term outputs. Always use MSE — CCC is unstable
		with near-constant predictions from auxiliary heads early in training."""
		if self.task == 'binary':
			return nn.BCEWithLogitsLoss()
		else:
			return nn.MSELoss()


	def _compute_metrics(self, predictions, labels):
		"""Return (primary_metric_value, metric_name) for logging."""
		if self.task == 'binary':
			probs = torch.sigmoid(predictions)
			preds_binary = (probs >= 0.5).float()
			correct = (preds_binary.view(-1) == labels.view(-1)).float()
			acc = correct.sum() / len(correct)
			return acc.item(), 'accuracy'
		else:
			corr = util.pearson_corr(predictions, labels)
			return corr, 'pearson_r'


	def train_model(self):

		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)

		epoch_start_time = time.time()
		min_loss = None

		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			term_name = name.split('_')[0]
			if '_direct_gene_layer.weight' in name:
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=False)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=True)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.lr)
		optimizer.zero_grad()

		if self.task == 'binary':
			print("epoch\ttrain_acc\ttrain_loss\tval_acc\tval_loss\tgrad_norm\telapsed_time")
		else:
			print("epoch\ttrain_corr\ttrain_loss\ttrue_auc\tpred_auc\tval_corr\tval_loss\tgrad_norm\telapsed_time")

		for epoch in range(self.data_wrapper.epochs):
			# Train
			self.model.train()
			train_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			_gradnorms = torch.zeros(len(train_loader)).cuda(self.data_wrapper.cuda)

			for i, (inputdata, labels) in enumerate(train_loader):
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				optimizer.zero_grad()

				aux_out_map,_ = self.model(cuda_features)

				if train_predict.size()[0] == 0:
					train_predict = aux_out_map['final'].data
					train_label_gpu = cuda_labels
				else:
					train_predict = torch.cat([train_predict, aux_out_map['final'].data], dim=0)
					train_label_gpu = torch.cat([train_label_gpu, cuda_labels], dim=0)

				total_loss = 0
				loss_fn = self._get_loss_fn()
				aux_loss_fn = self._get_aux_loss_fn()
				for name, output in aux_out_map.items():
					if name == 'final':
						total_loss += loss_fn(output, cuda_labels)
					else:
						aux_loss = aux_loss_fn(output, cuda_labels)
						if not torch.isnan(aux_loss):
							total_loss += self.data_wrapper.alpha * aux_loss

				if torch.is_tensor(total_loss) and not torch.isnan(total_loss):
					total_loss.backward()
				else:
					# Skip this batch if loss is NaN
					continue

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				# Clip gradients to prevent NaN propagation
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
				_gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0)
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0]
			train_metric, _ = self._compute_metrics(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			val_loss = 0

			for i, (inputdata, labels) in enumerate(val_loader):
				features = util.build_input_vector(inputdata, self.data_wrapper.cell_features)
				cuda_features = Variable(features.cuda(self.data_wrapper.cuda))
				cuda_labels = Variable(labels.cuda(self.data_wrapper.cuda))

				aux_out_map, _ = self.model(cuda_features)

				if val_predict.size()[0] == 0:
					val_predict = aux_out_map['final'].data
					val_label_gpu = cuda_labels
				else:
					val_predict = torch.cat([val_predict, aux_out_map['final'].data], dim=0)
					val_label_gpu = torch.cat([val_label_gpu, cuda_labels], dim=0)

				loss_fn = self._get_loss_fn()
				for name, output in aux_out_map.items():
					if name == 'final':
						val_loss += loss_fn(output, cuda_labels)

			val_metric, _ = self._compute_metrics(val_predict, val_label_gpu)

			epoch_end_time = time.time()

			if self.task == 'binary':
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, total_loss, val_metric, val_loss,
					gradnorms, epoch_end_time - epoch_start_time))
			else:
				true_auc = torch.mean(train_label_gpu)
				pred_auc = torch.mean(train_predict)
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, total_loss, true_auc, pred_auc,
					val_metric, val_loss, gradnorms, epoch_end_time - epoch_start_time))

			epoch_start_time = epoch_end_time

			if min_loss == None:
				min_loss = val_loss
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
			elif min_loss - val_loss > self.data_wrapper.delta:
				min_loss = val_loss
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))

		return min_loss