import numpy as np
np.Inf = np.inf
from executer_TP import Execute_TP
import pytorch_lightning as pl
import argparse
import os
from pytorch_lightning import Trainer, seed_everything
seed_everything(42, workers=True)

current_dir = os.getcwd()
DATA_PATH = os.path.join(current_dir,"data_TP")

def argparse_default(description=None):
    parser = pl.Trainer.add_argparse_args(argparse.ArgumentParser())

    #my editing
    # --- Model ablation flags (can be overridden by presets below) ---
    parser.add_argument("--loss_type", type=str, default="l1", choices=["l1", "huber"],
                        help="Loss for normalized endpoints: l1 or huber")
    parser.add_argument("--huber_beta", type=float, default=0.5,
                        help="Huber beta (only used if loss_type=huber)")
    parser.add_argument("--use_interaction", type=lambda s: str(s).lower() in ["1", "true", "yes"], default=False,
                        help="If true, add |h - t| to inputs")

    #parser.add_argument("--path_train_dataset", type=str, required=True, help="data_TP/dbpedia124k/")
    # Paths.
    parser.add_argument("--path_dataset_folder", type=str, default='data_TP/') #The folder path where your dataset is located

    parser.add_argument("--storage_path", type=str, default='HYBRID_Storage') #Location for storing model outputs.
    parser.add_argument("--eval_dataset", type=str, default='Dbpedia124k',
                        help="Available datasets: Dbpedia124k, Yago3K") #Specifies the dataset to use
    # FactBench, BPDP,Dbpedia34k,
    #TODO: To be added later for factbench dataset in particular
    parser.add_argument("--sub_dataset_path", type=str, default=None,
                        help="TODO: Available subpaths: bpdp/, domain/, domainrange/, mix/, property/, random/, range/,")

    #TODO: To be added later for factbench dataset in particular
    parser.add_argument("--prop", type=str, default=None,
                        help="TODO: Available properties (only for FactBench dataset if available): architect, artist, author, commander, director, musicComposer, producer, None")

    parser.add_argument("--negative_triple_generation", type=str, default="False",
                        help="Available approaches: corrupted-triple-based, corrupted-time-based, False")

    parser.add_argument("--complete_dataset", type=bool, default=True)
    parser.add_argument("--include_veracity", type=bool, default=True)

    # parser.add_argument("--auto_scale_batch_size", type=bool, default=True)
    # parser.add_argument("--deserialize_flag", type=str, default=None, help='Path of a folder for deserialization.')

    # Models. select temporal model for time point prediction!!
    parser.add_argument("--model", type=str, default='temporal-prediction-model',
                        help="Available models:temporal-prediction-model, temporal-full-hybrid") #Defines which model to use


    parser.add_argument("--task", type=str, default='time-prediction',
                        help="Available datasets:   time-prediction, fact-checking") #Specifies the task
                        # help="Available models:Hybrid, ConEx, TransE, Hybrid, ComplEx, RDF2Vec")

    parser.add_argument("--emb_type", type=str, default='dihedron',
                        help="Available TKG embeddings: dihedron, None")

    # Hyperparameters pertaining to number of parameters.
    parser.add_argument('--embedding_dim', type=int, default=100) #Hyperparameters. define the training setup
    parser.add_argument('--valid_ratio', type=int, default=20)
    parser.add_argument('--sentence_dim', type=int, default=768)
    parser.add_argument("--max_num_epochs", type=int, default=50) #Hyperparameters. define the training setup
    parser.add_argument("--min_num_epochs", type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512) #Hyperparameters. define the training setup
    parser.add_argument('--val_batch_size', type=int, default=1000)
    # parser.add_argument('--negative_sample_ratio', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of cpus used during batching')
    parser.add_argument("--check_val_every_n_epochs", type=int, default=10)
    # parser.add_argument('--enable_checkpointing', type=bool, default=True)
    # parser.add_argument('--deterministic', type=bool, default=True)
    # parser.add_argument('--fast_dev_run', type=bool, default=False)
    # parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    # PREPROCESS DATASETS
    parser.add_argument("--preprocess", type=str, default='False',
                        help="Available options: False, Concat, SentEmb, TrainTestTriplesCreate") #data preprocessing operations -> Concat for concatenating embeddings or SentEmb for sentence embeddings

    parser.add_argument("--ids_only", type=str, default=False)
    parser.add_argument("--checkpoint_dir_folder", type=str, default='2024-08-01 15:32:27.650994', choices=["all","YYYY-MM-DD HH:MM:SS.XXXXXX"], help="check hybrid storage folder")
    parser.add_argument(
        "--checkpoint_dataset_folder", default="dataset/", choices=["dataset/"], help="folder in which all resultant models are stored"
    ) #arguments define where the model checkpoints will be saved during training. The model's performance will also be evaluated periodically.

    if description is None:
        return parser.parse_args()
    else:
        return parser.parse_args(description)

if __name__ == '__main__':
    args = argparse_default()
    print("Parsed Arguments:", args)  # my editing for checking where my code gets killed?

    # ===== A/B/C/D PRESETS: uncomment ONE block you want to run =====

    # --- A: Baseline (L1, no |h-t|) ---
    #args.loss_type = "l1"
    #args.use_interaction = False
    # args.huber_beta = 0.5  # (ignored for L1)

    # --- B: L1 + |h-t| ---
    args.loss_type = "l1"
    args.use_interaction = True
    args.use_prod = False  # turn on h ⊙ r
    args.end_weight = 1.0  # asymmetric: weight end more
    args.extra_order_pen = 0.0  # small extra order penalty

    # --- C: Huber, no |h-t| ---
    #args.loss_type = "huber"
    #args.huber_beta = 0.5
    #args.use_interaction = False

    # --- D: Huber + |h-t| ---
    #args.loss_type = "huber"
    #args.huber_beta = 0.5
    #args.use_interaction = True
    # ================================================================

    exc = Execute_TP(args)
    exc.start()



    # Yago al
    # if args.eval_dataset == "Yago3K":
    #     args.ids_only = True

    # Preprocess dataset if flag is True!
    # if args.preprocess != 'False':
    #     if args.preprocess == 'Concat':
    #         if args.eval_dataset == "Dbpedia124k":
    #             DBpedia34kDataset = True
    #             # ConcatEmbeddings(args, path_dataset_folder=args.path_dataset_folder,DBpedia34k=DBpedia34kDataset)
    #         print("concat done")
    #     elif args.preprocess == 'SentEmb':
    #         print("sentence vectors creation is done")
    #     elif args.preprocess == 'TrainTestTriplesCreation':
    #         print("Train and Test triples creation is complete")

    # if args.eval_dataset == "Dbpedia124k" or args.eval_dataset == "Yago3K":
    # else:
    #     print("Please specify a valid dataset")
    #     exit(1)
