from dataclasses import fields

import pytest

from cli.shared.types.utils.device import DeviceConfig


@pytest.mark.cli
@pytest.mark.essential
def test_device_config_defaults():
    config = DeviceConfig()
    assert config.use_ddp == False
    assert config.processor == "cpu"
    assert config.gpu_ids == None


@pytest.mark.cli
@pytest.mark.essential
def test_device_config_custom_values():
    config = DeviceConfig(use_ddp=True, processor="gpu", gpu_ids=[0, 1])
    assert config.use_ddp == True
    assert config.processor == "gpu"
    assert config.gpu_ids == [0, 1]


@pytest.mark.cli
@pytest.mark.essential
def test_device_config_field_names():
    config_fields = {field.name for field in fields(DeviceConfig)}
    assert config_fields == {"use_ddp", "processor", "gpu_ids"}


@pytest.mark.cli
@pytest.mark.essential
def test_device_config_field_defaults():
    config_fields = {
        field.name: field.default for field in fields(DeviceConfig)}
    assert config_fields["use_ddp"] == False
    assert config_fields["processor"] == "cpu"
    assert config_fields["gpu_ids"] == None
