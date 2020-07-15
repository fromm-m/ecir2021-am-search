import logging

import torch

from arclus.data_split import DataSplit, OutputFeatures
from arclus.evaluation import class_metrics, f1_macro, accuracy
from arclus.experiment import Experiment
from arclus.n_end_model import NEndModel, set_random_seed, TrainTestHandler

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
    dataset = OutputFeatures()
    ds_info = dataset.info()
    split = DataSplit(dataset, shuffle=True)
    train_loader, val_loader, test_loader = split.get_train_val_test_split(batch_size=batch_size)

    base_model = NEndModel(
        input_shape=ds_info['feature_dim'],
        out=2,
        dropout_rate=dropout_rate,
    )
    optimizer = torch.optim.Adam(params=base_model.parameters())
    print(base_model)

    t_handler = TrainTestHandler(base_model=base_model, optimizer=optimizer)

    hyper_parameters = {
        'model_name': base_model.get_model_name(),
        'batch_size': batch_size,
        'random_seed': random_seed,
        'num_train_samples': len(split.train_indices),
        'num_test_samples': len(split.test_indices),
        'num_val_samples': len(split.val_indices),
        'task_name': 'SIM',
        'dropout_rate': dropout_rate,
    }
    if to_db:
        run_id, output_path = experiment.init_experiment(hyper_parameters=hyper_parameters)

    t_handler.train(train_loader, val_loader)

    true_labels_test = torch.tensor([], dtype=torch.long)
    pred_probs_test = torch.tensor([])
    for batch in test_loader:
        y = batch[1]
        x = batch[0]
        y_pred = t_handler.predict(x)
        pred_probs_test = torch.cat([pred_probs_test, y_pred])
        true_labels_test = torch.cat([true_labels_test, y])

    true_labels_val = torch.tensor([], dtype=torch.long)
    pred_probs_val = torch.tensor([])
    for batch in test_loader:
        y = batch[1]
        x = batch[0]
        y_pred = t_handler.predict(x)
        pred_probs_val = torch.cat([pred_probs_val, y_pred])
        true_labels_val = torch.cat([true_labels_val, y])

    test_acc = accuracy(pred_y=pred_probs_test, labels=true_labels_test, prob=True)
    test_f1_macro = f1_macro(pred_y=pred_probs_test, labels=true_labels_test, prob=True)
    metrics = class_metrics(pred_y=pred_probs_test, labels=true_labels_test, prob=True)

    val_acc = accuracy(pred_y=pred_probs_val, labels=true_labels_val, prob=True)
    val_f1_macro = f1_macro(pred_y=pred_probs_val, labels=true_labels_val, prob=True)
    val_metrics = class_metrics(pred_y=pred_probs_val, labels=true_labels_val, prob=True)

    print(test_acc)
    print(test_f1_macro)
    print('TEST RESULT')
    print(metrics)

    print('VALIDATION RESULT')
    print(val_metrics)

    print(f'n of epochs: {len(t_handler.history)}')

    result = {
        'test_acc': float(test_acc),
        'test_f1_macro': test_f1_macro,
        'val_acc': float(val_acc),
        'val_f1_macro': val_f1_macro,
        'val_loss': t_handler.history[-1],
        'number of epochs': len(t_handler.history)
    }
    if to_db:
        experiment.finalise_experiment(experiment_id=run_id, result=result)
