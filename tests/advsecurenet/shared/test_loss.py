import pytest
from torch import nn

from advsecurenet.shared.loss import Loss


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_loss_enum_values():
    assert Loss.CROSS_ENTROPY.value == nn.CrossEntropyLoss
    assert Loss.NLL_LOSS.value == nn.NLLLoss
    assert Loss.MARGIN_RANKING_LOSS.value == nn.MarginRankingLoss
    assert Loss.BCE_LOSS.value == nn.BCELoss
    assert Loss.BCE_WITH_LOGITS_LOSS.value == nn.BCEWithLogitsLoss
    assert Loss.HINGE_EMBEDDING_LOSS.value == nn.HingeEmbeddingLoss
    assert Loss.COSINE_EMBEDDING_LOSS.value == nn.CosineEmbeddingLoss
    assert Loss.MULTI_MARGIN_LOSS.value == nn.MultiMarginLoss
    assert Loss.SMOOTH_L1_LOSS.value == nn.SmoothL1Loss
    assert Loss.SOFT_MARGIN_LOSS.value == nn.SoftMarginLoss
    assert Loss.MULTI_LABEL_MARGIN_LOSS.value == nn.MultiLabelMarginLoss
    assert Loss.TRIPLET_MARGIN_LOSS.value == nn.TripletMarginLoss
    assert Loss.POISSON_NLL_LOSS.value == nn.PoissonNLLLoss
    assert Loss.KLDIV_LOSS.value == nn.KLDivLoss
    assert Loss.MSELoss.value == nn.MSELoss
    assert Loss.L1_LOSS.value == nn.L1Loss
    assert Loss.NLL_LOSS2D.value == nn.NLLLoss2d


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_loss_enum_names():
    assert Loss.CROSS_ENTROPY.name == "CROSS_ENTROPY"
    assert Loss.NLL_LOSS.name == "NLL_LOSS"
    assert Loss.MARGIN_RANKING_LOSS.name == "MARGIN_RANKING_LOSS"
    assert Loss.BCE_LOSS.name == "BCE_LOSS"
    assert Loss.BCE_WITH_LOGITS_LOSS.name == "BCE_WITH_LOGITS_LOSS"
    assert Loss.HINGE_EMBEDDING_LOSS.name == "HINGE_EMBEDDING_LOSS"
    assert Loss.COSINE_EMBEDDING_LOSS.name == "COSINE_EMBEDDING_LOSS"
    assert Loss.MULTI_MARGIN_LOSS.name == "MULTI_MARGIN_LOSS"
    assert Loss.SMOOTH_L1_LOSS.name == "SMOOTH_L1_LOSS"
    assert Loss.SOFT_MARGIN_LOSS.name == "SOFT_MARGIN_LOSS"
    assert Loss.MULTI_LABEL_MARGIN_LOSS.name == "MULTI_LABEL_MARGIN_LOSS"
    assert Loss.TRIPLET_MARGIN_LOSS.name == "TRIPLET_MARGIN_LOSS"
    assert Loss.POISSON_NLL_LOSS.name == "POISSON_NLL_LOSS"
    assert Loss.KLDIV_LOSS.name == "KLDIV_LOSS"
    assert Loss.MSELoss.name == "MSELoss"
    assert Loss.L1_LOSS.name == "L1_LOSS"
    assert Loss.NLL_LOSS2D.name == "NLL_LOSS2D"
