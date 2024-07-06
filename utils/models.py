import os
import sys
import torch
import warnings
import configparser
from typing import Tuple
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification)


class Models:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initializes models (tte_model_id) and tokenizer (tte_tokenizer) based on configuration (textConfig.ini).
        Sets device ("cuda:0" if GPU available, else "cpu").
        """
        if not self.__initialized:
            self.config = self.__get_config()
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.tte_model_id = self.config['TEXTEMOTIONS']['ModelId']
            (self.tte_tokenizer,
             self.tte_model) = self.__init_models(self.tte_model_id)

            self.__initialized = True

    def __get_config(self) -> configparser.ConfigParser:
        """
        Reads and returns the configuration from 'textConfig.ini', which contains
        necessary settings like model identifiers.
        Returns:
            configparser.ConfigParser: The loaded configuration settings.
        Raises:
            IOError: If the 'textConfig.ini' file is not present, the method raises an IOError and exits the program.
        """
        config = configparser.ConfigParser()
        if len(config.sections()) == 0:
            try:
                base_path = os.path.dirname(os.path.dirname(__file__))
                path = os.path.join(base_path, 'config', 'textConfig.ini')
                with open(path) as f:
                    config.read_file(f)
            except IOError:
                print("No file 'textConfig.ini' is present, the program can not continue")
                sys.exit()
        return config

    def __init_models(self, tte_model_id) -> Tuple:
        """
        Initializes and returns the tokenizer and model for text emotion analysis based on the specified model ID.
        Parameters:
            tte_model_id (str): Model identifier for loading tokenizer and model.
        Functionality:
            Initializes tokenizer and model for emotion analysis.
        """
        # Text to emotions
        tte_tokenizer = AutoTokenizer.from_pretrained(tte_model_id)
        tte_model = AutoModelForSequenceClassification.from_pretrained(tte_model_id)
        tte_model.to(self.device)

        return (tte_tokenizer,
                tte_model)
