import logging
import pprint

from sklearn.metrics import classification_report
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader

from preprocessing.dataset_reader import PrecomputedPairwiseFeatures, split
from arclus.evaluation import accuracy, f1_macro
from arclus.experiment import Experiment
from arclus.models.train_test_handler import TrainTestHandler
from arclus.utils import set_random_seed, flatten_dict

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
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
    )
    print(base_model)

    t_handler = TrainTestHandler(
        base_model=base_model,
        criterion=BCEWithLogitsLoss(),
        device=device,
    )

    hyper_parameters = {
        'model_name': 'simple_feedforward',
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
        for x, y in loader:
            y_pred = t_handler.predict(x)
            y_score.append(y_pred)
            y_true.append(y)
        y_score = torch.cat(y_score, dim=0)
        y_true = torch.cat(y_true, dim=0)

        y_pred = torch.round(torch.sigmoid(y_score))
        result[subset] = dict(
            accuracy=accuracy(pred_y=y_pred, labels=y_true),
            f1_macro=f1_macro(pred_y=y_pred, labels=y_true),
        )
        print(classification_report(y_pred=y_pred, y_true=y_true))

    result['num_epochs'] = len(t_handler.history)
    result['val_loss'] = t_handler.history[-1]
    pprint.pprint(result)

    if to_db:
        result = flatten_dict(result)
        experiment.finalise_experiment(experiment_id=run_id, result=result)
