import torch
from torch import nn
from torchvision.models import resnet18
from train import DatasetWrapper
from torch.utils.data import DataLoader
# use ignite for training this model
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine

# config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_EPOCHS = 1
BATCH_SIZE = 128
LOG_ITER = 75


# define resnet18 model
class R18(nn.Module):    
    def __init__(self):
        super(R18, self).__init__()

        self.model = resnet18(num_classes=10)

        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        return self.model(x)


# import the data
train_dataset_9less = torch.load('./dataset/MNIST/modified/train_dataset_9less.pt')
train_dataset_9 = torch.load('./dataset/MNIST/modified/train_dataset_9.pt')

# load the data
train_loader_9less = DataLoader(dataset=train_dataset_9less, batch_size=BATCH_SIZE, shuffle=True)
train_loader_9 = DataLoader(dataset=train_dataset_9, batch_size=BATCH_SIZE, shuffle=True)

# initialize model
model = R18().to(DEVICE)

# set up optmiizer and loss function
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

# now to create trainers and evaluators 
trainer = create_supervised_trainer(model, optimizer, criterion, DEVICE)

eval_metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}

train_evaluator = create_supervised_evaluator(model, metrics=eval_metrics, device=DEVICE)
#val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=DEVICE)

######## LOGGING ########
# intermediate results every 75 iterations
@trainer.on(Events.ITERATION_COMPLETED(every=LOG_ITER))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")

# results per epoch
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    train_evaluator.run(train_loader_9less)
    metrics = train_evaluator.state.metrics
    print(f"Training Results - Epoch[{trainer.state.epoch}] Avg accuracy: {metrics['accuracy']:.2f} Avg loss: {metrics['loss']:.2f}")

# Define a Tensorboard logger
tb_logger = TensorboardLogger(log_dir="./tb-logger")

# Attach handler to plot trainer's loss every 75 iterations
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED(every=LOG_ITER),
    tag="training",
    output_transform=lambda loss: {"batch_loss": loss},
)

# Attach handler for plotting both evaluators' metrics after every epoch completes  
tb_logger.attach_output_handler(
    train_evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="training",
    metric_names="all",
    global_step_transform=global_step_from_engine(trainer),
)

### run trainer ###
trainer.run(train_loader_9less, max_epochs=MAX_EPOCHS)

# close tensorboard logger
tb_logger.close()