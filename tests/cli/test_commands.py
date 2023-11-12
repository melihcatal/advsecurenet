import os
import pytest
from unittest.mock import patch, Mock
from click.testing import CliRunner
from requests import HTTPError

from cli.cli import download_weights, config_default, deepfool, fgsm, pgd, cw, lots, adversarial_training


class TestDownloadWeights:
    """
    Test the download_weights command.
    """

    def setup_method(self, method):
        self.runner = CliRunner()

    def teardown_method(self, method):
        pass

    def test_download_weights_success(self):
        with patch('cli.cli.util_download_weights', return_value=None) as mock_download:
            result = self.runner.invoke(
                download_weights, ['--model-name', 'resnet18', '--dataset-name', 'cifar10'])

            # Assertions
            assert result.exit_code == 0
            assert 'Downloaded weights to weights directory.' in result.output
            mock_download.assert_called_once_with(
                'resnet18', 'cifar10', None, None)

    def test_download_weights_file_exists(self):
        with patch('cli.cli.util_download_weights', side_effect=FileExistsError()) as mock_download:
            result = self.runner.invoke(
                download_weights, ['--model-name', 'resnet18', '--dataset-name', 'cifar10'])

            # Assertions
            assert result.exit_code == 0
            assert 'Model weights for resnet18 trained on cifar10 already exist' in result.output

    def test_download_weights_model_not_found(self):
        with patch('cli.cli.util_download_weights', side_effect=HTTPError()) as mock_download:
            result = self.runner.invoke(
                download_weights, ['--model-name', 'resnet18', '--dataset-name', 'cifar10'])

            # Assertions
            assert result.exit_code == 0
            assert 'Model weights for resnet18 trained on cifar10 not found' in result.output

    def test_download_weights_other_exception(self):
        with patch('cli.cli.util_download_weights', side_effect=Exception()) as mock_download:
            result = self.runner.invoke(
                download_weights, ['--model-name', 'resnet18', '--dataset-name', 'cifar10'])

            # Assertions
            assert result.exit_code == 0
            assert 'Error downloading model weights for resnet18 trained on cifar10!' in result.output


class TestAttackCommands:

    @pytest.fixture
    def runner(self):
        return CliRunner()

    # Parameterize the test to run for each attack command
    @pytest.mark.parametrize("attack_command", [deepfool, fgsm, pgd, cw, lots])
    def test_attack_basic(self, runner, attack_command):
        # Mock the execute_general_attack function
        with patch('cli.cli.execute_general_attack', return_value="Success!") as mock_attack:

            # Simulate invoking the command using the provided attack_command
            result = runner.invoke(attack_command)

            # Assertions
            assert result.exit_code == 0
            mock_attack.assert_called_once()


class TestConfigDefaultCommand:
    def setup_method(self):
        self.runner = CliRunner()

    def test_config_default_print(self):
        config_name = "train_config.yml"

        # Mock the inner methods
        mock_default_config = {"key1": "value1", "key2": "value2"}
        with patch("advsecurenet.utils.get_default_config_yml", return_value="/path/to/mock/config"), \
                patch("os.path.exists", return_value=True), \
                patch("advsecurenet.utils.config_utils.read_yml_file", return_value=mock_default_config):

            result = self.runner.invoke(
                config_default, ["--config-name", config_name, "--print-output"])

        # Assertions
        assert result.exit_code == 0
        assert f"Default configuration file for {config_name}:" in result.output
        assert "key1: value1" in result.output
        assert "key2: value2" in result.output

    def test_config_default_no_config_name(self):
        # Arrange
        result = self.runner.invoke(config_default, [])

        # Assertions
        assert isinstance(result.exception, ValueError)

    def test_config_default_file_not_found(self):
        config_name = "nonexistent_config.yml"

        # Mock the generate_default_config_yaml function to raise FileNotFoundError
        with patch("advsecurenet.utils.generate_default_config_yaml", side_effect=FileNotFoundError()):
            result = self.runner.invoke(
                config_default, ["--config-name", config_name])

        # Assertions
        assert f"Configuration file {config_name} not found!" in result.output


class TestAdversarialTrainingCommand:
    """
    Test the adversarial_training command.
    """

    def setup_method(self, method):
        self.runner = CliRunner()

    def test_adversarial_training_success(self):
        mocked_config_data = {'model': 'CustomMnistModel', 'models': [], 'attacks': [{'fgsm': [{'config': './fgsm_attack_config.yml'}]}], 'dataset_type': 'mnist', 'num_classes': 10, 'dataset_path': None, 'optimizer': 'adam', 'criterion': 'cross_entropy', 'epochs': 10, 'batch_size': 32, 'adv_coeff': 0.5, 'verbose': True, 'learning_rate': 0.001, 'momentum': 0.9,
                              'weight_decay': 0.0005, 'scheduler': None, 'scheduler_step_size': 10, 'scheduler_gamma': 0.1, 'num_workers': 4, 'device': 'CPU', 'save_model': True, 'save_model_path': None, 'save_model_name': None, 'save_checkpoint': False, 'save_checkpoint_path': None, 'save_checkpoint_name': None, 'checkpoint_interval': 1, 'load_checkpoint': False, 'load_checkpoint_path': None}
        with patch('cli.cli.load_configuration', return_value=mocked_config_data), \
                patch('cli.utils.adversarial_training_cli.AdversarialTrainingCLI') as mock_adversarial_training:
            mock_adversarial_training_instance = mock_adversarial_training.return_value
            mock_adversarial_training_instance.train.return_value = None

            result = self.runner.invoke(adversarial_training, [
                                        '--config', 'path/to/config.yml'])

            # Assertions
            assert result.exit_code == 0
            mock_adversarial_training.assert_called_once()
            mock_adversarial_training_instance.train.assert_called_once()

    def test_adversarial_training_no_config(self):
        result = self.runner.invoke(adversarial_training)

        # Assertions
        assert result.exit_code != 0
        assert "No configuration file provided for adversarial training!" in result.output

    def test_adversarial_training_config_not_found(self):
        with patch('cli.cli.load_configuration', side_effect=FileNotFoundError()) as mock_load_config:
            result = self.runner.invoke(adversarial_training, [
                                        '--config', 'path/to/nonexistent.yml'])

            # Assertions
            assert result.exit_code != 0
            assert "does not exist" in result.output
