import torch
from torch import nn
from torchvision.models import resnet18 # need implicitly to load R18 model
from train import DatasetWrapper # need this implicitly to load dataset
from resnet import R18
from torch.utils.data import DataLoader
from statistics import mean, stdev
from torcheval.metrics.functional import multiclass_f1_score
import os


# config
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu

def compare_models(directory, test_dataset):
    """
    This function will take in a directory that contains several resnet models and a test dataset to evaluate the model on.
    It then returns averages and stdev of certain metrics of the models compared together. 

    Args:
        directory (string): the directory that contains the models we want to compare. The models should be state_dicts
        test_dataset (dataset): the dataset which will be used to compare the models

    Returns:
        metrics (dict): a nested dict that contains the metrics calulated from the models compared. Loss and F1-Score here.
    """

    # load in the data and push them to the GPU
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))
    for inputs, labels in test_loader:
        inputs, labels  = inputs.to(DEVICE), labels.to(DEVICE)

    walker = os.walk(directory)
    criterion = nn.CrossEntropyLoss()

    losses = []
    f1_scores = []
    for _root, _dirs, models in walker:
        for model in models:
            # initialize and import the model
            r18 = R18()
            r18.load_state_dict(torch.load(directory+'/'+model))
            r18.eval()

            #  push model to device and calculate metrics 
            r18.to(DEVICE)
            output = r18(inputs)
            losses.append(criterion(output, labels).item())
            f1_scores.append(multiclass_f1_score(output, labels, num_classes=10).item())
            del output
            del r18

        avg_loss = mean(losses)
        stdev_loss = stdev(losses)
        
        avg_f1_score = mean(f1_scores)
        stdev_f1_score = stdev(f1_scores)

        del inputs, labels

        metrics = {
            'Loss' : {
                'Avg' : avg_loss,
                'Stdev' : stdev_loss,
            },
            'F1' : {
                'Avg' : avg_f1_score,
                'Stdev' : stdev_f1_score,
            }
        }

    return metrics

if __name__ == "__main__":
    # import the test set
    test_dataset = torch.load('./dataset/MNIST/modified/test_set.pt')

    # set up subdirectories to be walked over
    subdirectories = ['9less', 'all', 'all_norm', 'diverse_larger'] # need to add 'diverse when we get the data
    model_metrics = {}
    for dir in subdirectories:
        full_dir = "./MNIST_test_case/saved_models/" + dir
        model_metrics[dir] = compare_models(full_dir, test_dataset)


    # print results
    print('--------------------Losses------------------------')
    print(f"avg loss 9less: {model_metrics['9less']['Loss']['Avg']} +/- {model_metrics['9less']['Loss']['Stdev']}")
    print(f"avg loss all: {model_metrics['all']['Loss']['Avg']} +/- {model_metrics['all']['Loss']['Stdev']}")
    print(f"avg loss all_norm: {model_metrics['all_norm']['Loss']['Avg']} +/- {model_metrics['all_norm']['Loss']['Stdev']}")
    #print(f"avg loss diverse: {model_metrics['diverse']['Loss']['Avg']} +/- {model_metrics['diverse']['Loss']['Stdev']}")
    print(f"avg loss diverse_larger: {model_metrics['diverse_larger']['Loss']['Avg']} +/- {model_metrics['diverse_larger']['Loss']['Stdev']}")
    print('--------------------F1 Scores------------------------')
    print(f"avg f1 9less: {model_metrics['9less']['F1']['Avg']} +/- {model_metrics['9less']['F1']['Stdev']}")
    print(f"avg f1 all: {model_metrics['all']['F1']['Avg']} +/- {model_metrics['all']['F1']['Stdev']}")
    print(f"avg f1 all_norm: {model_metrics['all_norm']['F1']['Avg']} +/- {model_metrics['all_norm']['F1']['Stdev']}")
    #print(f"avg f1 diverse: {model_metrics['diverse']['F1']['Avg']} +/- {model_metrics['diverse']['F1']['Stdev']}")
    print(f"avg f1 diverse_larger: {model_metrics['diverse_larger']['F1']['Avg']} +/- {model_metrics['diverse_larger']['F1']['Stdev']}")
    print('--------------------------------------------')
    print(model_metrics)