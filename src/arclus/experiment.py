"""
Module for handling experiments.
"""
import datetime
import logging
import os

from typing import Dict, Any, Tuple

import torch
from mlflow.tracking import MlflowClient

from arclus.settings import DATA_EXPERIMENTS


class Experiment:
    """Base class for initializing and finalizing experiments."""

    def __init__(
        self,
        root=DATA_EXPERIMENTS,
        tracking_uri="http://mlflow.dbs.ifi.lmu.de:5000",
        exp_name="Test_ARCLUS"
    ):
        self.root = root
        self.tracking_uri = tracking_uri
        self.exp_name = exp_name
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = self.client.get_experiment_by_name(self.exp_name).experiment_id
        self.root = root

    def init_experiment(
        self,
        hyper_parameters: Dict[str, Any],
    ) -> Tuple[str, str]:
        """
        Initialising and experiment. Creates an experiment run and creates an output directory based on the run id.
        :param hyper_parameters: The hyperparameters for this experiment to save in DB.
        :return: The id of the experiment run and the created output path.
        """
        current_run = self.client.create_run(self.experiment_id)
        self.client.log_param(run_id=current_run.info.run_id, key="create_time_utc",
                              value=datetime.datetime.now(tz=datetime.timezone.utc))

        for key in hyper_parameters.keys():
            self.client.log_param(run_id=current_run.info.run_id, key=key, value=hyper_parameters[key])

        # create output directory
        logging.debug('Create experiment directory: {}')
        output_path = os.path.join(self.root, str(current_run.info.run_id))
        if os.path.exists(output_path):
            logging.error('Output path already exists! {p}'.format(p=output_path))
        os.mkdir(output_path)

        self.client.log_param(run_id=current_run.info.run_id, key="out_dir", value=output_path)
        # return id as experiment handle, and path of existing directory to store outputs to
        return current_run.info.run_id, output_path

    def finalise_experiment(
        self,
        experiment_id: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Finalise the experiment by saving high-level results in DB.
        :param experiment_id: The id of the run to finalise.
        :param result: The result metrics to save.
        """
        for key in result.keys():
            self.client.log_metric(run_id=experiment_id, key=key, value=result[key])

    @staticmethod
    def save_model(
        path: str,
        model: torch.nn.Module
    ):
        """
        Save a torch model to given output path.
        :param path: The path.
        :param model: The model to save.
        """
        data_path = os.path.join(path, 'checkpoint.pth.tar')
        torch.save(model, os.path.join(path, 'model.pth'))

        torch.save({
            'model_state_dict': model.base.model.state_dict(),
            'optimizer_state_dict': model.base.optimizer.state_dict(),
        }, data_path)
