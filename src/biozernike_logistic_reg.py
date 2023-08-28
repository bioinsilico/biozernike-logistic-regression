import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.logistic_regr_nn import LogisticRegression
from src.biozernike_data_set import BiozernikeDataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

learningRate = 1e-5
epochs = 1000
batch_size = 2 ** 8
l2_weight = 1

cath_coefficients_file = "../resources/cath_moments.tsv"
ecod_coefficients_file = "../resources/ecod_moments.tsv"

dataset = BiozernikeDataset(cath_coefficients_file)
weights = dataset.weights()
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

testing_set = BiozernikeDataset(ecod_coefficients_file)
test_dataloader = DataLoader(testing_set, batch_size=len(testing_set), shuffle=True)
x_test, y_test = next(iter(test_dataloader))

writer = SummaryWriter()

model = LogisticRegression(input_features=3922)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

batch_n = 1
print("Starting training. Number of batches: %s" % (len(train_dataloader)))
for epoch in range(epochs):
    loss = None
    for x_train, y_train in train_dataloader:
        optimizer.zero_grad()
        y_predicted = model(x_train)
        loss = criterion(y_predicted, y_train) + l2_weight * torch.sum(model.get_weights() ** 2)
        loss.backward()
        optimizer.step()

        if batch_n % 1000 == 0:
            y_evaluation = model(x_test)
            roc_auc = roc_auc_score(y_test.detach().numpy(), y_evaluation.detach().numpy())
            precision, recall, thresholds = precision_recall_curve(
                y_test.detach().numpy(),
                y_evaluation.detach().numpy()
            )
            pr_auc = auc(recall, precision)
            print("epoch %s, batch number %s, loss %s, testing roc %s auc %s" % (
                epoch,
                batch_n,
                loss.item(),
                roc_auc,
                pr_auc
            ))
            writer.add_scalar("Loss/1000batches", loss, batch_n//1000)
            writer.add_scalar("ROC/1000batches", roc_auc, batch_n//1000)
            writer.add_scalar("AUC/1000batches", pr_auc, batch_n//1000)
            writer.add_histogram("Weights/1000batches", model.get_weights(), batch_n//1000)
            writer.flush()

        batch_n += 1

writer.close()
