import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as du
from contextlib import nullcontext
from torch.autograd import Variable

import util
from training_data_wrapper import *
from drugcell_nn import *


def _render_nn_architecture(dG, term_size_map, term_direct_gene_map, root, out_path):
    """Render the ontology DAG (= NN architecture) as a PNG and save to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import networkx as nx

    # Assign each node a layer via longest-path from root (top-down BFS)
    layer = {root: 0}
    for node in nx.topological_sort(dG):
        for child in dG.successors(node):
            layer[child] = max(layer.get(child, 0), layer[node] + 1)

    max_layer = max(layer.values()) if layer else 0
    nodes_by_layer = {}
    for node, lyr in layer.items():
        nodes_by_layer.setdefault(lyr, []).append(node)

    # Assign x positions within each layer
    pos = {}
    for lyr, nodes in nodes_by_layer.items():
        nodes_sorted = sorted(nodes)
        width = len(nodes_sorted)
        for i, node in enumerate(nodes_sorted):
            pos[node] = ((i - (width - 1) / 2.0), -lyr)

    # Node color by number of directly annotated genes
    direct_counts = [len(term_direct_gene_map.get(n, [])) for n in dG.nodes()]
    max_count = max(direct_counts) if direct_counts else 1
    max_count = max_count or 1
    node_colors = [cm.YlOrRd(len(term_direct_gene_map.get(n, [])) / max_count) for n in dG.nodes()]

    # Node size by total annotated genes (term_size_map)
    max_size = max(term_size_map.values()) if term_size_map else 1
    node_sizes = [20 + 180 * (term_size_map.get(n, 1) / max_size) for n in dG.nodes()]

    num_nodes = len(dG.nodes())
    fig_w = max(12, num_nodes * 0.15)
    fig_h = max(6, (max_layer + 1) * 0.8)
    fig, ax = plt.subplots(figsize=(min(fig_w, 40), min(fig_h, 24)))

    nx.draw(
        dG, pos=pos, ax=ax,
        node_color=node_colors, node_size=node_sizes,
        edge_color="#aaaaaa", arrows=True, arrowsize=6,
        with_labels=False, alpha=0.85,
    )

    sm = plt.cm.ScalarMappable(cmap=cm.YlOrRd, norm=plt.Normalize(0, max_count))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label="Direct gene annotations")

    ax.set_title(
        f"NeST-VNN Architecture  |  {num_nodes} terms  |  root: {root}",
        fontsize=11, pad=8,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


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

		mlflow_enabled = getattr(self.data_wrapper, 'mlflow_enabled', False)
		if mlflow_enabled:
			try:
				import mlflow as _mlflow
				from pathlib import Path
				_mlflow.set_experiment("nest_vnn")
				ctx = _mlflow.start_run()
				# Persist run_id so predict step can link back to this run
				run_id_path = Path(self.data_wrapper.modeldir) / "mlflow_run_id.txt"
				run_id_path.write_text(ctx.info.run_id)
			except ImportError:
				print("Warning: mlflow not installed; disabling MLflow logging.")
				mlflow_enabled = False
				ctx = nullcontext()
		else:
			ctx = nullcontext()

		with ctx:
			return self._train_model_inner(mlflow_enabled)


	def _train_model_inner(self, mlflow_enabled):
		if mlflow_enabled:
			import mlflow, json
			from pathlib import Path
			from mlflow.data.http_dataset_source import HTTPDatasetSource
			from mlflow.data.meta_dataset import MetaDataset
			dw = self.data_wrapper
			# Derive study_id and read metadata written by cbioport_transform.py
			train_path = Path(dw.train)
			ndir = train_path.parent          # .../nest_vnn_input/
			study_id = ndir.parent.name       # .../output/<study_id>/nest_vnn_input
			meta_path = ndir / "metadata.json"
			meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

			params = {
				"study_id":            meta.get("study_id", study_id),
				"label_col":           dw.label_col,
				"task":                dw.task,
				"lr":                  dw.lr,
				"wd":                  dw.wd,
				"alpha":               dw.alpha,
				"epochs":              dw.epochs,
				"batchsize":           dw.batchsize,
				"num_hiddens_genotype": dw.num_hiddens_genotype,
				"dropout_fraction":    dw.dropout_fraction,
				"min_dropout_layer":   dw.min_dropout_layer,
				"zscore_method":       dw.zscore_method,
				"patience":            dw.patience,
				"delta":               dw.delta,
				"num_terms":           len(dw.dG.nodes()),
				"num_genes":           len(dw.gene_id_mapping),
				"num_train_samples":   len(self.train_feature),
				"num_val_samples":     len(self.val_feature),
			}
			if meta.get("ndex_uuid"):
				params["ndex_uuid"] = meta["ndex_uuid"]
			if meta.get("min_alt_freq") is not None:
				params["min_alt_freq"] = meta["min_alt_freq"]
			if meta.get("gene_count"):
				params["gene_count"] = meta["gene_count"]
			mlflow.log_params(params)
			mlflow.set_tags({
				"study_id": meta.get("study_id", study_id),
				"task":     dw.task,
				"label":    dw.label_col,
			})

			# Log cBioPortal study and NDEx hierarchy as dataset inputs
			if meta.get("cbioportal_url"):
				src = HTTPDatasetSource(url=meta["cbioportal_url"])
				mlflow.log_input(MetaDataset(source=src, name=study_id), context="cbioportal_study")
			if meta.get("ndex_url"):
				src = HTTPDatasetSource(url=meta["ndex_url"])
				mlflow.log_input(MetaDataset(source=src, name=meta.get("ndex_uuid", "ontology")), context="ontology")

		self.model = DrugCellNN(self.data_wrapper)
		self.model.cuda(self.data_wrapper.cuda)

		if mlflow_enabled:
			import mlflow
			from pathlib import Path
			dw = self.data_wrapper
			arch_png = Path(dw.modeldir) / "nn_architecture.png"
			try:
				_render_nn_architecture(
					dw.dG, dw.term_size_map, dw.term_direct_gene_map, dw.root, str(arch_png)
				)
				mlflow.log_artifact(str(arch_png), artifact_path="architecture")
			except Exception as e:
				print(f"Warning: could not render NN architecture image: {e}")

		epoch_start_time = time.time()
		min_loss = None
		best_val_metric = None
		best_train_metric = None
		best_val_loss = None
		best_train_loss = None
		best_epoch = None

		early_stopping_counter = 0
		term_mask_map = util.create_term_mask(self.model.term_direct_gene_map, self.model.gene_dim, self.data_wrapper.cuda)
		for name, param in self.model.named_parameters():
			if '_direct_gene_layer.weight' in name:
				term_name = name.split('_direct_gene_layer')[0]
				param.data = torch.mul(param.data, term_mask_map[term_name]) * 0.1
			else:
				param.data = param.data * 0.1

		train_loader = du.DataLoader(du.TensorDataset(self.train_feature, self.train_label), batch_size=self.data_wrapper.batchsize, shuffle=True, drop_last=False)
		val_loader = du.DataLoader(du.TensorDataset(self.val_feature, self.val_label), batch_size=self.data_wrapper.batchsize, shuffle=False)

		optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.data_wrapper.lr, betas=(0.9, 0.99), eps=1e-05, weight_decay=self.data_wrapper.wd)
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
			epoch_train_loss = 0.0
			n_train_batches = 0

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
					epoch_train_loss += total_loss.item()
					n_train_batches += 1
				else:
					# Skip this batch if loss is NaN
					continue

				for name, param in self.model.named_parameters():
					if '_direct_gene_layer.weight' not in name:
						continue
					term_name = name.split('_direct_gene_layer')[0]
					param.grad.data = torch.mul(param.grad.data, term_mask_map[term_name])

				# Clip gradients to prevent NaN propagation
				torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
				_gradnorms[i] = util.get_grad_norm(self.model.parameters(), 2.0).unsqueeze(0)
				optimizer.step()

			gradnorms = sum(_gradnorms).unsqueeze(0).cpu().numpy()[0]
			epoch_train_loss = epoch_train_loss / max(n_train_batches, 1)
			if train_predict.size()[0] == 0:
				train_metric = float('nan')
			else:
				train_metric, _ = self._compute_metrics(train_predict, train_label_gpu)

			self.model.eval()

			val_predict = torch.zeros(0, 0).cuda(self.data_wrapper.cuda)
			epoch_val_loss = 0.0
			n_val_batches = 0

			with torch.no_grad():
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
							epoch_val_loss += loss_fn(output, cuda_labels).item()
							n_val_batches += 1

			epoch_val_loss = epoch_val_loss / max(n_val_batches, 1)
			val_metric, metric_name = self._compute_metrics(val_predict, val_label_gpu)

			epoch_end_time = time.time()

			if self.task == 'binary':
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, epoch_train_loss, val_metric, epoch_val_loss,
					gradnorms, epoch_end_time - epoch_start_time))
			else:
				true_auc = float(torch.mean(train_label_gpu)) if train_predict.size()[0] > 0 else float('nan')
				pred_auc = float(torch.mean(train_predict)) if train_predict.size()[0] > 0 else float('nan')
				print("{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
					epoch, train_metric, epoch_train_loss, true_auc, pred_auc,
					val_metric, epoch_val_loss, gradnorms, epoch_end_time - epoch_start_time))

			epoch_start_time = epoch_end_time

			if min_loss is None or epoch_val_loss < min_loss - self.data_wrapper.delta:
				min_loss = epoch_val_loss
				best_val_metric = val_metric
				best_train_metric = train_metric
				best_val_loss = epoch_val_loss
				best_train_loss = epoch_train_loss
				best_epoch = epoch
				early_stopping_counter = 0
				torch.save(self.model, self.data_wrapper.modeldir + '/model_final.pt')
				print("Model saved at epoch {}".format(epoch))
			else:
				early_stopping_counter += 1
				if early_stopping_counter >= self.data_wrapper.patience:
					print("Early stopping at epoch {}".format(epoch))
					break

			if mlflow_enabled:
				import mlflow
				metrics = {
					f"train_{metric_name}": train_metric,
					"train_loss": epoch_train_loss,
					f"val_{metric_name}": val_metric,
					"val_loss": epoch_val_loss,
					"grad_norm": float(gradnorms),
				}
				if best_val_metric is not None:
					metrics[f"best_val_{metric_name}"] = best_val_metric
					metrics[f"best_train_{metric_name}"] = best_train_metric
					metrics["best_val_loss"] = best_val_loss
					metrics["best_train_loss"] = best_train_loss
					metrics["best_epoch"] = best_epoch
				mlflow.log_metrics(metrics, step=epoch)

		if mlflow_enabled:
			import mlflow
			mlflow.log_artifact(self.data_wrapper.modeldir + '/model_final.pt')
			mlflow.log_artifact(self.data_wrapper.std)

		return min_loss