from enum import Enum


class CLITrainerErrorMessages(Enum):
    """
    Contains all error messages for the CLI Trainer.
    """
    NORM_LAYER_MISSING_MEAN_OR_STD = "Please provide mean and std for normalization layer!"
    NORM_LAYER_MEAN_OR_STD_NOT_LIST = "The mean and std should be a list of floats!"
    NORM_LAYER_LENGTH_MISMATCH_MEAN_AND_STD = "The mean and std should have the same length!"
    NORM_LAYER_LENGTH_MISMATCH_MEAN_AND_NUM_INPUT_CHANNELS = "The length of the mean should be equal to the number of input channels!"
    NORM_LAYER_LENGTH_MISMATCH_STD_AND_NUM_INPUT_CHANNELS = "The length of the std should be equal to the number of input channels!"


class CLIErrorMessages(Enum):
    """
    Contains all error messages for the CLI.
    """
    TRAINER = CLITrainerErrorMessages
