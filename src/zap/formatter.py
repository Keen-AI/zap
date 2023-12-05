
import logging
import warnings

from colorama import Fore, Style, init


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': f"{Fore.RED}{Style.BRIGHT}",
    }

    def format(self, record):
        log_message = super().format(record)
        colour = self.COLORS.get(record.levelname, Fore.RESET)

        level = f"{colour+Style.BRIGHT}[{record.levelname}]{Style.RESET_ALL}"
        return f"{level} {colour}{log_message}"


class CustomWarningHandler:
    def __init__(self):
        self.logger = logging.getLogger('lightning')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())
        self.logger.handlers[0].setFormatter(ColorFormatter())

    def write(self, message, *args, **kwargs):
        self.logger.warning(message)


def supress_pydantic_warnings():
    # supress Pydantic UserWarnings entirely
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


def format_lightning_warnings_and_logs():
    # initialise colorama
    init(autoreset=True)

    #  remove Lightning's existing stream handler
    logger = logging.getLogger('lightning')
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # set the custom warning handler to print out formatted messages
    warnings.showwarning = CustomWarningHandler().write
