import logging
import pprint

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader

from arclus.data.data_split import PrecomputedPairwiseFeatures, split
from arclus.evaluation import accuracy, class_metrics, f1_macro
from arclus.experiment import Experiment
from arclus.models.train_test_handler import TrainTestHandler
from arclus.utils import set_random_seed

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    to_db = False

    random_seed = 1
    set_random_seed(random_seed)

    experiment = Experiment()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1000
    dropout_rate = 0.1
    # TODO: hyperparameter search?

    # prepare dataset split
    dataset = PrecomputedPairwiseFeatures()
    ds_info = dataset.info()
    train_loader, val_loader, test_loader = [
        DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        for dataset in split(dataset=dataset)
    ]

    base_model = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features=ds_info['feature_dim'], out_features=1, bias=True),
    ).to(device=device)
    print(base_model)
    optimizer = torch.optim.Adam(params=base_model.parameters())

    t_handler = TrainTestHandler(base_model=base_model, optimizer=optimizer)

    hyper_parameters = {
        'model_name': base_model.get_model_name(),
        'batch_size': batch_size,
        'random_seed': random_seed,
        'num_train_samples': len(train_loader),
        'num_test_samples': len(test_loader),
        'num_val_samples': len(val_loader),
        'task_name': 'SIM',
        'dropout_rate': dropout_rate,
    }
    t_handler.train(train_loader, val_loader)

    if to_db:
        run_id, output_path = experiment.init_experiment(hyper_parameters=hyper_parameters)
        experiment.save_model(output_path, t_handler.model, t_handler.optimizer)

    result = dict()
    for subset, loader in dict(
        test=test_loader,
        val=val_loader,
    ).items():
        y_true = []
        y_score = []
        for x, y in test_loader:
            y_pred = t_handler.predict(x)
            y_score.append(y_pred)
            y_true.append(y)
        y_score = torch.cat(y_score, dim=0)
        y_true = torch.cat(y_true, dim=0)

        result[subset] = dict(
            accuary=accuracy(pred_y=y_score, labels=y_true, prob=True),
            f1_macro=f1_macro(pred_y=y_score, labels=y_true, prob=True),
            metrics=class_metrics(pred_y=y_score, labels=y_true, prob=True),
        )

    result['num_epochs'] = len(t_handler.history)
    result['val_loss'] = t_handler.history[-1]
    pprint.pprint(result)

    if to_db:
        experiment.finalise_experiment(experiment_id=run_id, result=result)
