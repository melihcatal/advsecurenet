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
        with patch("advsecurenet.utils.config_utils.get_default_config_yml", return_value="/path/to/mock/config"), \
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
        with patch("advsecurenet.utils.config_utils.generate_default_config_yaml", side_effect=FileNotFoundError()):
            result = self.runner.invoke(
                config_default, ["--config-name", config_name])

        # Assertions
        assert f"Configuration file {config_name} not found!" in result.output


class TestAdversarialTrainingCommand:
    """
    Test the adversarial_training command.
    """

    @pytest.fixture
    def runner(self):
        return CliRunner()

    @pytest.fixture
    def set_working_dir(self):
        # Save the original working directory
        original_dir = os.getcwd()
        # Get the path to the 'tests/cli' directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        print("current_dir: ", current_dir)
        cli_dir = os.path.join(current_dir, '..')
        os.chdir(current_dir)
        yield
        # Reset the working directory after the test
        os.chdir(original_dir)

    # @patch('cli.utils.config.load_configuration')

    @patch('cli.cli.AdversarialTrainingCLI')
    def test_adversarial_training_with_valid_config(self, mock_adversarial_training_cli, runner, set_working_dir):
        # Mock the AdversarialTrainingCLI class
        mock_adversarial_training_cli.return_value = Mock(
            adversarial_training=Mock(return_value=None))

        # Simulate invoking the command
        result = runner.invoke(adversarial_training, [
                               '--config', './adversarial_training_config.yml'])

        # Assertions
        assert result.exit_code == 0
        mock_adversarial_training_cli.assert_called_once()

    @patch('cli.cli.load_configuration')
    def test_adversarial_training_without_config(self, mock_load_config, runner):
        mock_load_config.side_effect = Exception(
            "No configuration file provided for adversarial training")
        result = runner.invoke(adversarial_training)
        assert result.exit_code != 0
        assert "No configuration file provided for adversarial training" in result.output
        mock_load_config.assert_not_called()

    @patch('cli.cli.load_configuration')
    def test_adversarial_training_with_invalid_config(self, mock_load_config, runner):
        mock_load_config.side_effect = Exception("Invalid configuration file")
        result = runner.invoke(adversarial_training, [
                               '--config', 'invalid/path.yml'])
        assert result.exit_code != 0
        assert "does not exist" in result.output
        # assert not called
        mock_load_config.assert_not_called()
