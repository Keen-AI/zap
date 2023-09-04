import os

from dotenv import load_dotenv

load_dotenv()


def get_env(x, default=None, allow_none=False):
    var = os.environ.get(x, default)
    if not var and not allow_none:
        raise EnvironmentError(f'Missing environment variable {x}. This variable is not allowed to be null.')
    return var


def parse_null_num(x, type_):
    if x:
        return type_(x)
    return None


class Config():
    def __init__(self) -> None:
        self.arch = get_env('ARCH')
        self.bias = get_env('BIAS')  # not sure if the bias is needed when not training

        # TODO: we don't need this for inference yet it raises an error if not present
        self.label_column = get_env('LABEL_COLUMN')

        self.batch_size = int(get_env('BATCH_SIZE'))
        self.image_w = int(get_env('IMAGE_W'))
        self.image_h = int(get_env('IMAGE_H'))
        self.pretrained = get_env('PRETRAINED')
        self.label_map_path = get_env('LABEL_MAP_PATH')


class TrainConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.train_image_path = get_env('TRAIN_IMAGE_PATH')
        self.train_labels_path = get_env('TRAIN_LABELS_PATH')

        self.val_image_path = get_env('VAL_IMAGE_PATH')
        self.val_labels_path = get_env('VAL_LABELS_PATH')

        self.learning_rate = float(get_env('LEARNING_RATE'))
        self.momentum = float(get_env('MOMENTUM'))
        self.max_epochs = int(get_env('MAX_EPOCHS'))

        # TODO: convert to int, handle nulls
        self.train_limit = parse_null_num(get_env('TRAIN_LIMIT', allow_none=True), int)
        self.val_limit = parse_null_num(get_env('VAL_LIMIT', allow_none=True), int)

        self.validate_training_data = get_env('VALIDATE_TRAINING_DATA', allow_none=True)
        self.validate_validation_data = get_env('VALIDATE_VALIDATION_DATA', allow_none=True)


class TestConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.experiment_id = get_env('TEST_EXPERIMENT_ID')
        self.checkpoint_file = get_env('TEST_CHECKPOINT_FILE')
        self.test_image_path = get_env('TEST_IMAGE_PATH')
        self.test_labels_path = get_env('TEST_LABELS_PATH')

        self.test_limit = parse_null_num(get_env('TEST_LIMIT', allow_none=True), int)
        self.validate_testing_data = get_env('VALIDATE_TESTING_DATA', allow_none=True)


class InferrenceConfig(Config):
    def __init__(self) -> None:
        super().__init__()
        self.experiment_id = get_env('INFER_EXPERIMENT_ID')
        self.checkpoint_file = get_env('INFER_CHECKPOINT_FILE')
        self.images_path = get_env('IMAGES_PATH')
