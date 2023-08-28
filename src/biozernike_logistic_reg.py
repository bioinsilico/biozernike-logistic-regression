import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.logistic_regr_nn import LogisticRegression
from src.biozernike_data_set import BiozernikeDataset

from sklearn.metrics import roc_auc_score

learningRate = 1e-5
epochs = 100
batch_size = 2**8
l2_weight = 1

cath_coefficients_file = "../resources/cath_moments.tsv"
ecod_coefficients_file = "../resources/ecod_moments.tsv"

dataset = BiozernikeDataset(cath_coefficients_file)
weights = dataset.weights()
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

testing_set = BiozernikeDataset(ecod_coefficients_file)
x_test = torch.stack([x for x, y in testing_set], dim=0)
y_test = torch.stack([y for x, y in testing_set], dim=0)

model = LogisticRegression(input_features=3919)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

print("Starting training. Number of batches: %s" % (len(train_dataloader)))
for epoch in range(epochs):
    loss = None
    batch_n = 1
    for x_train, y_train in train_dataloader:
        optimizer.zero_grad()
        y_predicted = model(x_train)
        loss = criterion(y_predicted, y_train) + l2_weight * torch.sum(model.get_weights()**2)
        loss.backward()
        optimizer.step()

        y_evaluation = model(x_test)
        auc = roc_auc_score(y_test.detach().numpy(), y_evaluation.detach().numpy())
        if batch_n % 100 == 0:
            print("epoch %s, batch number %s, loss %s, testing auc %s" % (epoch, batch_n, loss.item(), auc))
        batch_n += 1
