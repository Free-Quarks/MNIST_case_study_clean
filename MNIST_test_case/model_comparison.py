import torch
from torch import nn
from torchvision.models import resnet18 # need implicitly to load R18 model
from train import DatasetWrapper # need this implicitly to load dataset
from resnet import R18
from torch.utils.data import DataLoader
from statistics import mean, stdev


# config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 'cuda:0' if only one gpu

# import the test set
test_dataset = torch.load('./dataset/MNIST/modified/test_set.pt')
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset))

# import the models and set up the loss criterion, we annoyingly need to instatiate it multiple times
model_9less_0 = R18()
model_9less_1 = R18()
model_9less_2 = R18()
model_9less_3 = R18()
model_all_0 = R18()
model_all_1 = R18()
model_all_2 = R18()
model_all_3 = R18()
model_all_norm_0 = R18()
model_all_norm_1 = R18()
model_all_norm_2 = R18()
model_all_norm_3 = R18()
#model_diverse_0 = R18()
#model_diverse_1 = R18()
#model_diverse_2 = R18()
#model_diverse_3 = R18()
model_diverse_larger_0 = R18()
model_diverse_larger_1 = R18()
model_diverse_larger_2 = R18()
model_diverse_larger_3 = R18()

criterion = nn.CrossEntropyLoss()

# 9less data models
model_9less_0.load_state_dict(torch.load("./MNIST_test_case/saved_models/9less/9less_0.pt"))
model_9less_1.load_state_dict(torch.load("./MNIST_test_case/saved_models/9less/9less_1.pt"))
model_9less_2.load_state_dict(torch.load("./MNIST_test_case/saved_models/9less/9less_2.pt"))
model_9less_3.load_state_dict(torch.load("./MNIST_test_case/saved_models/9less/9less_3.pt"))
model_9less_0.eval()
model_9less_1.eval()
model_9less_2.eval()
model_9less_3.eval()

# all data models
model_all_0.load_state_dict(torch.load("./MNIST_test_case/saved_models/all/all_0.pt"))
model_all_1.load_state_dict(torch.load("./MNIST_test_case/saved_models/all/all_1.pt"))
model_all_2.load_state_dict(torch.load("./MNIST_test_case/saved_models/all/all_2.pt"))
model_all_3.load_state_dict(torch.load("./MNIST_test_case/saved_models/all/all_3.pt"))
model_all_0.eval()
model_all_1.eval()
model_all_2.eval()
model_all_3.eval()

# all normalized data amount models
model_all_norm_0.load_state_dict(torch.load("./MNIST_test_case/saved_models/all_norm/all_norm_0.pt"))
model_all_norm_1.load_state_dict(torch.load("./MNIST_test_case/saved_models/all_norm/all_norm_1.pt"))
model_all_norm_2.load_state_dict(torch.load("./MNIST_test_case/saved_models/all_norm/all_norm_2.pt"))
model_all_norm_3.load_state_dict(torch.load("./MNIST_test_case/saved_models/all_norm/all_norm_3.pt"))
model_all_norm_0.eval()
model_all_norm_1.eval()
model_all_norm_2.eval()
model_all_norm_3.eval()

# diverse normalized data models
"""
model_diverse_0.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse/diverse_0.pt"))
model_diverse_1.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse/diverse_1.pt"))
model_diverse_2.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse/diverse_2.pt"))
model_diverse_3.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse/diverse_3.pt"))
model_diverse_0.eval()
model_diverse_1.eval()
model_diverse_2.eval()
model_diverse_3.eval()
"""

# diverse data models
model_diverse_larger_0.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse_larger/diverse_larger_0.pt"))
model_diverse_larger_1.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse_larger/diverse_larger_1.pt"))
model_diverse_larger_2.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse_larger/diverse_larger_2.pt"))
model_diverse_larger_3.load_state_dict(torch.load("./MNIST_test_case/saved_models/diverse_larger/diverse_larger_3.pt"))
model_diverse_larger_0.eval()
model_diverse_larger_1.eval()
model_diverse_larger_2.eval()
model_diverse_larger_3.eval()

# now to compute the loss on the test set, Test set should be batched as a single batch 
for inputs, labels in test_loader:
    inputs, labels  = inputs.to(DEVICE), labels.to(DEVICE)

# initialize lists
loss_9less = []
loss_all = []
loss_all_norm = []
#loss_diverse = []
loss_diverse_larger = []

# compute all the outputs, need to load and unload models and outputs for memory considerations

###### 9less models ######
model_9less_0.to(DEVICE)
output = model_9less_0(inputs)
loss_9less.append(criterion(output, labels).item())
del output

model_9less_1.to(DEVICE)
output = model_9less_1(inputs)
loss_9less.append(criterion(output, labels).item())
del output

model_9less_2.to(DEVICE)
output = model_9less_2(inputs)
loss_9less.append(criterion(output, labels).item())
del output

model_9less_3.to(DEVICE)
output = model_9less_3(inputs)
loss_9less.append(criterion(output, labels).item())
del output

# compute statistics here so gc can clear memory
avg_loss_9less = mean(loss_9less)
stdev_loss_9less = stdev(loss_9less)

###### all models ######
model_all_0.to(DEVICE)
output = model_all_0(inputs)
loss_all.append(criterion(output, labels).item())
del output

model_all_1.to(DEVICE)
output = model_all_1(inputs)
loss_all.append(criterion(output, labels).item())
del output

model_all_2.to(DEVICE)
output = model_all_2(inputs)
loss_all.append(criterion(output, labels).item())
del output

model_all_3.to(DEVICE)
output = model_all_3(inputs)
loss_all.append(criterion(output, labels).item())
del output

# compute statistics here so gc can clear memory
avg_loss_all = mean(loss_all)
stdev_loss_all = stdev(loss_all)

###### all_norm models ######
model_all_norm_0.to(DEVICE)
output = model_all_norm_0(inputs)
loss_all_norm.append(criterion(output, labels).item())
del output

model_all_norm_1.to(DEVICE)
output = model_all_norm_1(inputs)
loss_all_norm.append(criterion(output, labels).item())
del output

model_all_norm_2.to(DEVICE)
output = model_all_norm_2(inputs)
loss_all_norm.append(criterion(output, labels).item())
del output

model_all_norm_3.to(DEVICE)
output = model_all_norm_3(inputs)
loss_all_norm.append(criterion(output, labels).item())
del output

# compute statistics here so gc can clear memory
avg_loss_all_norm = mean(loss_all_norm)
stdev_loss_all_norm = stdev(loss_all_norm)

"""
###### diverse models ######
model_diverse_0.to(DEVICE)
output = model_diverse_0(inputs)
loss_diverse.append(criterion(output, labels).item())
del output

model_diverse_1.to(DEVICE)
output = model_diverse_1(inputs)
loss_diverse.append(criterion(output, labels).item())
del output

model_diverse_2.to(DEVICE)
output = model_diverse_2(inputs)
loss_diverse.append(criterion(output, labels).item())
del output

model_diverse_3.to(DEVICE)
output = model_diverse_3(inputs)
loss_diverse.append(criterion(output, labels).item())
del output

# compute statistics here so gc can clear memory
avg_loss_diverse = mean(loss_diverse)
stdev_loss_diverse = stdev(loss_diverse)
"""

###### diverse_larger models ######
model_diverse_larger_0.to(DEVICE)
output = model_diverse_larger_0(inputs)
loss_diverse_larger.append(criterion(output, labels).item())
del output

model_diverse_larger_1.to(DEVICE)
output = model_diverse_larger_1(inputs)
loss_diverse_larger.append(criterion(output, labels).item())
del output

model_diverse_larger_2.to(DEVICE)
output = model_diverse_larger_2(inputs)
loss_diverse_larger.append(criterion(output, labels).item())
del output

model_diverse_larger_3.to(DEVICE)
output = model_diverse_larger_3(inputs)
loss_diverse_larger.append(criterion(output, labels).item())
del output

# compute statistics here so gc can clear memory
avg_loss_diverse_larger = mean(loss_diverse_larger)
stdev_loss_diverse_larger = stdev(loss_diverse_larger)


# print results
print(f"avg loss 9less: {avg_loss_9less} +/- {stdev_loss_9less}")
print(f"avg loss all: {avg_loss_all} +/- {stdev_loss_all}")
print(f"avg loss all_norm: {avg_loss_all_norm} +/- {stdev_loss_all_norm}")
#print(f"avg loss diverse: {avg_loss_diverse} +/- {stdev_loss_diverse}")
print(f"avg loss diverse_larger: {avg_loss_diverse_larger} +/- {stdev_loss_diverse_larger}")