from enum import Enum
from torch import nn

class Loss(Enum):
    CROSS_ENTROPY = nn.CrossEntropyLoss
    NLL_LOSS = nn.NLLLoss
    MARGIN_RANKING_LOSS = nn.MarginRankingLoss
    BCE_LOSS = nn.BCELoss
    BCE_WITH_LOGITS_LOSS = nn.BCEWithLogitsLoss
    HINGE_EMBEDDING_LOSS = nn.HingeEmbeddingLoss
    COSINE_EMBEDDING_LOSS = nn.CosineEmbeddingLoss
    MULTI_MARGIN_LOSS = nn.MultiMarginLoss
    SMOOTH_L1_LOSS = nn.SmoothL1Loss
    SOFT_MARGIN_LOSS = nn.SoftMarginLoss
    MULTI_LABEL_MARGIN_LOSS = nn.MultiLabelMarginLoss
    TRIPLET_MARGIN_LOSS = nn.TripletMarginLoss
    POISSON_NLL_LOSS = nn.PoissonNLLLoss
    KLDIV_LOSS = nn.KLDivLoss
    MSELoss = nn.MSELoss
    L1_LOSS = nn.L1Loss
    NLL_LOSS2D = nn.NLLLoss2d