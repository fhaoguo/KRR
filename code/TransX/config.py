# import sys
# sys.path.append(".")

import torch
from utils.utils import CheckPath


class Config():
    def __init__(self):
        # Data arguments
        self.pos_path = "../../data/TransX/train.txt"
        self.valid_path = "../../data/TransX/valid.txt"
        self.test_path = "../../data/TransX/valid.txt"
        self.ent_path = "../../data/TransX/entity_dict.json"
        self.rel_path = "../../data/TransX/relation_dict.json"
        self.embed_path = "../../save/TransX/embed/"
        self.log_path = "../../save/TransX/log/"
        self.save_type = "pkl"

        # Dataloader arguments
        self.batch_size = 1024
        self.shuffle = True
        self.num_workers = 0
        self.drop_last = False
        self.rep_proba = 0.5
        self.ex_proba = 0.5

        # Model and training general arguments
        self.TransE = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2}
        self.TransH = {"EmbeddingDim": 100,
                       "Margin":       1.0,
                       "L":            2,
                       "C":            0.01,
                       "Eps":          0.001}
        self.TransD = {"EntityDim":    100,
                       "RelationDim":  100,
                       "Margin":       2.0,
                       "L":            2}
        self.TransA = {"EmbeddingDim": 100,
                       "Margin":       3.2,
                       "L":            2,
                       "Lamb":         0.01,
                       "C":            0.2}
        self.KG2E   = {"EmbeddingDim": 100,
                       "Margin":       4.0,
                       "Sim":          "EL",
                       "Vmin":         0.03,
                       "Vmax":         3.0}

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_name = "TransE"
        self.weight_decay = 0
        self.epochs = 5
        self.eval_epoch = 1
        self.learning_rate = 0.01
        self.lr_decay = 0.8
        self.lr_decay_epoch = 5
        self.optimizer = "Adam"
        self.eval_method = "MR"
        self.sim_measure = "L2"
        self.model_save = "param"
        self.model_path = "../../save/TransX/model/"
        self.load_embed = False
        self.entity_file = "../../save/TransX/embed/entityEmbedding.txt"
        self.relation_file = "../../save/TransX/embed/relationEmbedding.txt"
        self.pre_model = "../../save/TransX/model/TransE_ent128_rel128.param"

        # Other arguments
        self.summary_dir = "../../save/TransX/log/TransE/"

        # Check Path
        self.CheckPath()

        # self.usePaperConfig()

    def usePaperConfig(self):
        # Paper best params
        if self.model_name == "TransE":
            self.embeddingdim = 50
            self.learning_rate = 0.01
            self.margin = 1.0
            self.distance = 1
            self.sim_measure = "L1"
        elif self.model_name == "TransH":
            self.batch_size = 1200
            self.embeddingdim = 50
            self.learning_rate = 0.005
            self.margin = 0.5
            self.C = 0.015625
        elif self.model_name == "TransD":
            self.batch_size = 4800
            self.entitydim = 100
            self.relationdim = 100
            self.margin = 2.0

    def CheckPath(self):
        # Check files
        CheckPath(self.pos_path)
        CheckPath(self.valid_path)

        # Check dirs
        CheckPath(self.model_path, raise_error=False)
        CheckPath(self.summary_dir, raise_error=False)
        CheckPath(self.log_path, raise_error=False)
        CheckPath(self.embed_path, raise_error=False)
