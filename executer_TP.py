import warnings

import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping

from sklearn.metrics import accuracy_score, classification_report, precision_score, auc
import nn_models_TP
from utils_TP.dataset_classes import StandardDataModule
from data_TP import Data
from utils_TP.static_funcs import *
import time
import torch
from sklearn.model_selection import KFold
# from pytorch_lightning_kfold.validation import KFoldCrossValidator

import json
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin,DataParallelPlugin
# from utils.dataset_classes import StandardDataModule
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)


class Execute_TP:
    def __init__(self, args):
        args = preprocesses_input_args(args)
        sanity_checking_with_arguments(args)
        self.args = args
        # 1. Create an instance of KG.
        self.args.dataset = Data(args=args)
        print(f"[DEBUG] idx_time_dict keys: {list(self.args.dataset.idx_time_dict.keys())[:10]}")  # Print first 10 keys
        self.args.idx_time_dict = self.args.dataset.idx_time_dict  # Add this line to store the dictionary

        # 2. Create a storage path  + Serialize dataset object.
        self.storage_path = create_experiment_folder(folder_name=args.storage_path)
        # self.eval_model = True if self.args.eval == 1 else False

        # 3. Save Create folder to serialize data. This two numerical value will be used in embedding initialization.
        self.args.num_entities, self.args.num_relations, self.args.num_times = self.args.dataset.num_entities, self.args.dataset.num_relations, self.args.dataset.num_times


        # 4. Create logger
        self.logger = create_logger(name=self.args.model, p=self.storage_path)

        # 5. KGE related parameters


    def store(self, trained_model) -> None:
        """
        Store trained_model model and save embeddings into csv file.
        :param trained_model:
        :return:
        """
        self.logger.info('Store full model.')
        # Save Torch model.
        torch.save(trained_model.state_dict(), self.storage_path + '/model.pt')
        self.args.dataset = ""
        with open(self.storage_path + '/configuration.json', 'w') as file_descriptor:
            temp = vars(self.args)
            temp.pop('gpus')
            temp.pop('tpu_cores')
            json.dump(temp, file_descriptor)

        self.logger.info('Stored data.')


    def start(self) -> None:
        """
        Train and/or Evaluate Model
        Store Mode
        """
        print("Training Started...")    # my editing for checking where my code gets killed?
        start_time = time.time()
        # 1. Train and Evaluate
        trained_model = self.train_and_eval()
        print("Training and evaluation Complete.")      # my editing for checking where my code gets killed?
        # 2. Store trained model
        self.store(trained_model)
        #
        total_runtime = time.time() - start_time

        print(f"Total Runtime: {total_runtime} seconds")        # my editing for checking where my code gets killed?

        if 60 * 60 > total_runtime:
            message = f'{total_runtime / 60:.3f} minutes'
        else:
            message = f'{total_runtime / (60 ** 2):.3f} hours'

        self.logger.info(f'Total Runtime:{message}')

    def train_and_eval(self) -> nn_models_TP.BaseKGE:
        """
        Training and evaluation procedure
        """
        print("Training and evaluation Started...")         # my editing for checking where my code gets killed?

        self.logger.info('--- Parameters are parsed for training ---')


        # trainer = pl.Trainer.from_argparse_args(Namespace(**dict(train_config)), early_stop_callback=early_stop_callback)

        # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
        # self.args.enable_checkpointing = True
        self.args.checkpoint_callback = True
        if self.args.sub_dataset_path==None:
            pth = self.args.eval_dataset
        else:
            pth = self.args.eval_dataset + "-" + self.args.sub_dataset_path.replace('/','')
        mdl = self.args.model

        # saves a checkpint model file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.storage_path,
            filename="sample-{"+(str(mdl).lower())+"}--{"+(str(pth).lower())+"}--"+(str(self.args.emb_type).lower())+"--"+(str(self.args.negative_triple_generation).lower())+"--{epoch:02d}-{val_loss:.3f}",
            save_top_k=1,
            mode="min",
        )
        early_stopping_callback = EarlyStopping(monitor="val_loss", patience=10)
        # 1. Create Pytorch-lightning Trainer object from input configuration
        # print(torch.cuda.device_count())
        if torch.cuda.is_available():
            self.trainer = pl.Trainer.from_argparse_args(self.args, plugins=DataParallelPlugin(),
                                                     callbacks = [early_stopping_callback, checkpoint], gpus=1)
        else:
            self.trainer = pl.Trainer.from_argparse_args(self.args,
                                                     callbacks = [early_stopping_callback, checkpoint])
        # 2. Check whether validation and test datasets are available.
        #if self.args.dataset.is_valid_test_available():
        trained_model = self.training()
        print("Training fininshed")         # my editing for checking where my code gets killed?
        self.logger.info('--- Training is completed  ---')

        # print(self.args.checkpoint_callback.best_model_path)

        return trained_model

    def training(self):
        """
        Train models with KvsAll or NegativeSampling
        :return:
        """
        # 1. Select model and labelling : triple prediction.
        print("Loading data")       # my editing for checking where my code gets killed?

        model, form_of_labelling = select_model(self.args)
        if not self.args.batch_size:
            self.args.batch_size = int(len(self.args.dataset.idx_train_set) / 3) + 1
        if not self.args.val_batch_size:
            self.args.val_batch_size = int(len(self.args.dataset.idx_valid_set) / 2) + 1

        self.args.fast_dev_run=False
        self.args.accumulate_grad_batches = self.args.batch_size
        self.args.deterministic=True

        self.logger.info(f' Standard training starts: {model.name}-labeling:{form_of_labelling}')

        print("  >> #valid triples:", len(self.args.dataset.idx_valid_set))

      #  print(" Execute_TP sees dataset.paired_train_idx", getattr(self.args.dataset, "paired_train_idx", None))
        # 2. Create training data.
        dataset = StandardDataModule(train_set_idx=self.args.dataset.idx_train_set,
                                     valid_set_idx=self.args.dataset.idx_valid_set,
                                     test_set_idx=self.args.dataset.idx_test_set,
                                     entities_count=self.args.dataset.num_entities,
                                     relations_count=self.args.dataset.num_relations,
                                     times_count=self.args.dataset.num_times,
                                     form=form_of_labelling,
                                     batch_size=self.args.batch_size,
                                     num_workers=self.args.num_workers)

        print("Data loaded successfully")       # my editing for checking where my code gets killed?

        # 3. Display the selected model's architecture.
        self.logger.info(model)

        train_data = dataset.train_dataloader(batch_size1=self.args.batch_size)
        batch = next(iter(train_data))
        print(f"[DEBUG] range batch sampe:", batch)
        val_data = dataset.val_dataloader(batch_size1=self.args.val_batch_size)

     #   print(f"Training Data Size: {len(train_data.dataset)}")         # my editing for checking where my code gets killed?
     #   print(f"Validation Data Size: {len(val_data.dataset)}")         # my editing for checking where my code gets killed?

        # Create a KFoldCrossValidator instance
        # validator = KFoldCrossValidator(model, train_data, val_data, k=5)
        # self.trainer.add_callback(validator)
        # 5. Train model
        self.trainer.fit(model, train_data,val_data)
        # 6. Test model on validation and test sets if possible.
        #if self.args.task != 'range-prediction':
        #self.trainer.test(ckpt_path='best',test_dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_set)))
        test_results = self.trainer.test(ckpt_path='best',
                                         dataloaders=dataset.dataloaders(len(self.args.dataset.idx_test_set)))
        test_loss = float(test_results[0].get("test_loss", float("nan")))
        #self.evaluate(model, dataset.train_set_idx, 'Evaluation of Train data: ' + form_of_labelling)
        #self.evaluate(model, dataset.test_set_idx, 'Evaluation of Test data: ' + form_of_labelling)
        if self.args.task == 'range-prediction':
            train_metrics = self.evaluate(model, self.args.dataset.idx_train_set, 'Evaluation of Train data: ' + form_of_labelling)
            test_metrics  = self.evaluate(model, self.args.dataset.idx_test_set, 'Evaluation of Test data: '+ form_of_labelling)

            #FOR PUTTING EVERYTHING IN THE CSV file
            # Also log best val/test losses if you want
            # You can read the last logged values from self.trainer.callback_metrics if needed
            from pathlib import Path, PurePath
            import csv, datetime

            results_path = Path(self.storage_path) / "results.csv"
            header = [
                "timestamp", "run_folder", "preset", "loss_type", "huber_beta", "use_interaction",
                "order_penalty_lambda", "embedding_dim", "batch_size", "lr",
                "val_loss_best", "test_loss",
                "train_exact_match", "train_mae_start", "train_mae_end",
                "test_exact_match", "test_mae_start", "test_mae_end"
            ]

            # pull some values
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            run_folder = self.storage_path
            loss_type = getattr(self.args, "loss_type", "l1")
            huber_beta = getattr(self.args, "huber_beta", 0.5)
            use_interaction = getattr(self.args, "use_interaction", False)
            order_penalty_lambda = getattr(self.args, "order_penalty_lambda", 0.0)
            embedding_dim = self.args.embedding_dim
            batch_size = self.args.batch_size
            lr = 1e-3  # or read from model if you prefer

            # read from checkpoint filename or callback metrics if available
            #val_loss_best = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float('nan'))))
            #val_loss_best = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float("nan"))))
            # After self.trainer.test(...), test loss is in self.trainer.callback_metrics as well
            #test_loss = float(self.trainer.callback_metrics.get("test_loss", torch.tensor(float('nan'))))

           # row = [
            #    ts, run_folder, loss_type, huber_beta, use_interaction,
            #    order_penalty_lambda, embedding_dim, batch_size, lr,
            #    val_loss_best, test_loss,
           #     train_metrics["exact_match"], train_metrics["mae_start_years"], train_metrics["mae_end_years"],
            #    test_metrics["exact_match"], test_metrics["mae_start_years"], test_metrics["mae_end_years"],
           # ]

            # best val_loss from checkpoint callback (more accurate than last logged)
            best_val = None
            for cb in self.trainer.callbacks:
                if hasattr(cb, "best_model_score") and cb.best_model_score is not None:
                    best_val = cb.best_model_score
            val_loss_best = float(best_val.item()) if best_val is not None else float("nan")
# test_loss was captured earlier from self.trainer.test() return value
# already stored in `test_loss` variable

# pull actual learning rate from model (default fallback if missing)
            lr = getattr(model, "lr", 1e-4)
# optional: preset label for convenience
            preset = f"{loss_type}_{'int' if use_interaction else 'base'}"

            row = [
                ts, run_folder, preset, loss_type, huber_beta, use_interaction,
                order_penalty_lambda, embedding_dim, batch_size, lr,
                val_loss_best, test_loss,
                train_metrics["exact_match"], train_metrics["mae_start_years"], train_metrics["mae_end_years"],
                test_metrics["exact_match"], test_metrics["mae_start_years"], test_metrics["mae_end_years"],
            ]
            file_exists = results_path.exists()
            with open(results_path, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(header)
                w.writerow(row)
            # FOR PUTTING EVERYTHING IN THE CSV file

        else:
            self.evaluate(model, self.args.dataset.idx_train_set, 'Evaluation of Train data: ' + form_of_labelling)
            self.evaluate(model, self.args.dataset.idx_test_set, 'Evaluation of Test data: '+ form_of_labelling)
        return model

    def mrr_score2(self, predictions, labels):
        # Convert predictions and labels to numpy arrays
        predictions = np.array(predictions)
        labels = np.array(labels)

        # Compute the reciprocal rank for each query
        reciprocal_ranks = []
        for query_index in range(len(predictions)):
            # Get the prediction and label for the current query
            prediction = predictions[query_index]
            label = labels[query_index]

            # Find the rank of the highest ranked relevant item
            rank = np.where(prediction == label)[0][0] + 1
            reciprocal_rank = 1.0 / rank
            reciprocal_ranks.append(reciprocal_rank)

        # Return the mean of all the reciprocal ranks
        return np.mean(reciprocal_ranks)
    def mrr_score(self, y_true, y_pred):
        """
        Calculate MRR (Mean Reciprocal Rank) for a list of predictions.

        Parameters:
        y_true (array): An array of true target values.
        y_pred (array): An array of predicted target values.

        Returns:
        float: The MRR score.
        """
        ranks = []
        for yt, yp in zip(y_true, y_pred):
            rank = np.where(yp == yt)[0][0] + 1
            ranks.append(1 / rank)

        return np.mean(ranks)
    def evaluate(self, model, triple_idx, info):
        dev = next(model.parameters()).device
        print("evaluation")
        model.eval()
        self.logger.info(info)
        #self.logger.info(f'Num of triples {len(triple_idx)}')
        self.logger.info(f'Num of records {len(triple_idx)}')
        """
        X_test = np.array(triple_idx)[:, :6]
        y_test = np.array(triple_idx)[:, -1]

        # label = model.time_embeddings(y_test)
        label = y_test
        X_test_tensor = torch.Tensor(X_test).long()
        Y_test_tensor = torch.Tensor(y_test).long()
        idx_s, idx_p, idx_o, t_idx, s_idx, v_data = X_test_tensor[:, 0], X_test_tensor[:, 1], X_test_tensor[:, 2], X_test_tensor[:, 3], X_test_tensor[:, 4], X_test_tensor[:, 5]
        # 2. Prediction score
        if info.__contains__("Test"):
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data,type="test")
        else:
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data)
        """
        #load into a single tensor
        X = np.array(triple_idx)
        X_tensor = torch.LongTensor(X)
        if self.args.task == 'range-prediction':
            #5 column input: (h, r, t, y1, y2)
            print(f"[DEBUG] eval input shape (Nx5): {X_tensor.shape}")
            idx_s, idx_p, idx_o, y1_idx, y2_idx = (X_tensor[:, i] for i in range(5))
            prob = model.forward_triples(idx_s, idx_p, idx_o, y1_idx, y2_idx, type="test" if "Test" in info else None)
            start_pred, end_pred = prob
            pred_start = start_pred.round().long().clamp(min=0, max=self.args.num_times-1)
            pred_end = end_pred.round().long().clamp(min=0, max=self.args.num_times - 1)
            correct = ((pred_start == y1_idx) & (pred_end == y2_idx)).float().mean()
            print(f"[INFO] Eval {info!r} exact-match accuracy: {correct*100:.2f}%")

            # Add MAE in years
            y1_years = torch.tensor([float(model.year_idx_dict[int(i.item())]) for i in y1_idx], device=dev)
            y2_years = torch.tensor([float(model.year_idx_dict[int(i.item())]) for i in y2_idx], device=dev)

            pred_start_years = torch.tensor([float(model.year_idx_dict[int(p.item())]) for p in pred_start],
                                            device=dev)
            pred_end_years = torch.tensor([float(model.year_idx_dict[int(p.item())]) for p in pred_end],
                                          device=dev)
            mae_start = torch.abs(pred_start_years - y1_years).float().mean()
            mae_end = torch.abs(pred_end_years - y2_years).float().mean()
            print(f"[INFO] Eval {info!r} MAE (start year): {mae_start:.2f} years")
            print(f"[INFO] Eval {info!r} MAE (end year): {mae_end:.2f} years")

            # NEW: ±k accuracy
            for k in (1, 3, 5, 10):
                acc_k = (((pred_start_years - y1_years).abs() <= k) &
                         ((pred_end_years - y2_years).abs() <= k)).float().mean()
                print(f"[INFO] Eval {info!r} ±{k}y accuracy: {acc_k * 100:.2f}%")

            # NEW: interval IoU (inclusive years; drop the +1 if you prefer continuous years)
            inter_left = torch.max(pred_start_years, y1_years)
            inter_right = torch.min(pred_end_years, y2_years)
            inter = (inter_right - inter_left + 1).clamp(min=0)
            union = (torch.max(pred_end_years, y2_years) - torch.min(pred_start_years, y1_years) + 1)
            iou = (inter / union).mean()
            print(f"[INFO] Eval {info!r} Interval IoU: {iou:.3f}")

            #  return a dict so callers can log it
            return {
                "exact_match": float((correct * 100.0).item()),
                "mae_start_years": float(mae_start.item()),
                "mae_end_years": float(mae_end.item()),
            }
        else:
            #original 6 column path: (h, r, t, time, sent_idx, veracity)
            X6 = X_tensor[:, :6]
            #y = torch.LongTensor(X[:, -1]).long()
            idx_s, idx_p, idx_o, t_idx, s_idx, v_data = (X6[:, i] for i in range(6))
            prob = model.forward_triples(idx_s, idx_p, idx_o, t_idx, s_idx, v_data, type="test" if "Test" in info else None)
        # pred = (prob > 0.5).float()
            pred = prob.data.detach().numpy()
            max_pred = np.argmax(pred, axis=1)
            idx, sort_pred= torch.sort(prob,dim=1,descending=True)
            return None

        # test_mrr = self.mrr_score(label, sort_pred)
        # self.logger.info(test_mrr)
        # self.logger.info( accuracy_score(max_pred, label))
        # self.logger.info(classification_report(max_pred, label))


# true negatives are ignored
