import pytest
import enum
from torch import nn
from unittest.mock import patch
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.models.custom_model import CustomModel
from advsecurenet.models.standard_model import StandardModel
from advsecurenet.shared.types import ModelType

model_variants = ["resnet18", "vgg16"]


def test_infer_model_type_standard():
    model_type = ModelFactory.infer_model_type('resnet18')
    assert model_type == ModelType.STANDARD


def test_infer_model_type_custom():
    model_type = ModelFactory.infer_model_type('CustomMnistModel')
    assert model_type == ModelType.CUSTOM


def test_infer_model_type_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.infer_model_type('unsupported_model')


def test_get_model_standard():
    model = ModelFactory.get_model('resnet18', num_classes=10)
    assert model is not None


def test_get_model_custom():
    model = ModelFactory.get_model('CustomMnistModel', num_classes=10)
    assert model is not None


def test_get_model_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.get_model('unsupported_model', num_classes=10)


def test_available_models():
    models = ModelFactory.available_models()
    assert len(models) > 0


def test_get_available_standard_models():
    models = ModelFactory.available_standard_models()
    assert len(models) > 0


def test_get_available_custom_models():
    models = ModelFactory.available_custom_models()
    assert len(models) > 0


def test_available_model_weights():
    weights = ModelFactory.available_weights("resnet18")
    assert type(weights) == enum.EnumMeta
    assert len(weights) > 0


def test_available_model_weights_unsupported():
    with pytest.raises(ValueError):
        ModelFactory.available_weights("unsupported_model")


def test_available_model_weights_custom():
    with patch('advsecurenet.models.model_factory.ModelFactory.infer_model_type', return_value=ModelType.CUSTOM):
        # expect value error
        with pytest.raises(ValueError, match="Custom models do not support pretrained weights."):
            ModelFactory.available_weights("CustomMnistModel")


def test_custom_model_weights():
    with pytest.raises(NotImplementedError):
        CustomModel.available_weights("CustomMnistModel")


def test_standard_model_weights():
    weights = StandardModel.available_weights("resnet18")
    assert type(weights) == enum.EnumMeta
    assert len(weights) > 0


def test_standard_model_modify_model_resnet():
    model = StandardModel(model_variant="resnet18",
                          num_classes=10, pretrained=False)
    assert model.model.conv1.in_channels == 3
    assert model.model.fc.out_features == 10


def test_standard_model_modify_model_vgg16():
    model = StandardModel(model_variant="vgg16",
                          num_classes=10, pretrained=False)
    assert model.model.features[0].in_channels == 3
    assert model.model.classifier[6].out_features == 10


def test_standard_model_modify_model_pretrained():
    model = StandardModel(model_variant="resnet18",
                          num_classes=10, pretrained=True)
    assert model.model.conv1.in_channels == 3
    assert model.model.fc.out_features == 10


def test_standard_model_modify_model_pretrained_weights():
    model = StandardModel(model_variant="resnet18", num_classes=10,
                          pretrained=True, weights="IMAGENET1K_V1")
    assert model.model.conv1.in_channels == 3
    assert model.model.fc.out_features == 10


def test_get_layer_names():
    model = StandardModel(model_variant="resnet18", num_classes=10)
    layer_names = model.get_layer_names()
    assert len(layer_names) > 0


@pytest.mark.parametrize("model_variant", model_variants)
def test_get_layer(model_variant):
    model = StandardModel(model_variant=model_variant, num_classes=10)
    target_layer = model.get_layer_names()[0]
    layer = model.get_layer(target_layer)
    assert layer is not None


def test_set_layer():
    model = StandardModel(model_variant="resnet18", num_classes=10)
    old_layer = model.get_layer("conv1")
    assert old_layer is not None
    assert old_layer.in_channels == 3
    new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                          padding=3, bias=False)
    model.set_layer("conv1", new_layer)
    updated_layer = model.get_layer("conv1")
    assert updated_layer is not None
    assert updated_layer == new_layer
