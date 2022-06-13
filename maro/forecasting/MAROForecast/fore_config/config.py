import os
import datetime
import logging
import shutil
import yaml
import torch
import pandas as pd
class dict_to_obj(object):
    def __init__(self, input_dict):
        for name, value in input_dict.items():
            if isinstance(value, (list, tuple)):
                value_list = []
                for value_item in value: 
                    if isinstance(value_item, dict):
                        value_list.append(dict_to_obj(value_item))
                    else:
                        value_list.append(value_item)
                setattr(self, name, value_list)
            elif isinstance(value, dict):
                setattr(self, name, dict_to_obj(value))
            elif isinstance(value, datetime.date):
                setattr(self, name, pd.to_datetime(value))
            else:
                setattr(self, name, value)

# Global config helper 
class Config(object):
    def __init__(self, config_path):
        self.load_config(config_path)
        self.init_experiment()
        self.generate_logger()
        self.init_device()
        self.logger.info("Init from %s." % config_path)
        self.logger.info("Output dir %s." % self.output_dir)
        if self.experiment.backup_code:
            self.save_code()
    
    def load_config(self, config_file_path):
        with open(config_file_path, 'r') as f:
            yaml_config = yaml.full_load(f)
        for name, value in vars(dict_to_obj(yaml_config)).items():
            setattr(self, name, value)

    def init_experiment(self):
        time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(self.experiment.output_dir, self.experiment.experiment_name + "_" + time)
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.log_dir = os.path.join(self.output_dir, 'log')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def generate_logger(self):
        log_formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)-8s: %(message)s")
        logging.getLogger().setLevel(0)
        self.logger = logging.getLogger(self.experiment.experiment_name)
        file_handler = logging.FileHandler(os.path.join(self.log_dir , "log.txt"))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(0)
        self.logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(0)
        self.logger.addHandler(console_handler)
    
    def init_device(self):
        self.experiment.device = torch.device(self.experiment.device if torch.cuda.is_available() else "cpu")

    def save_code(self):
        self.logger.info("Saving code")
        code_dir = "."
        self.code_backup_dir = os.path.join(self.output_dir, 'code_snapshot')
        os.makedirs(self.code_backup_dir, exist_ok=True)
        for root, _, files in os.walk(code_dir, topdown=False):
            if (len(root.split(os.path.sep)) < 2 or root.split(os.path.sep)[1] != self.output_dir.split(os.path.sep)[-2]) and root.split(os.path.sep)[-1][:2] != "__":
                os.makedirs(os.path.join(self.code_backup_dir, root), exist_ok=True)
                for name in files:
                    file_path = os.path.join(root, name)
                    output_file_path = os.path.join(self.code_backup_dir, root, name)
                    shutil.copyfile(file_path, output_file_path)
        self.logger.info("Code saved to %s" % self.code_backup_dir)