import torch
import time
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.logistic_regr_nn import LogisticRegression, LogisticRegressionUniform
from src.biozernike_data_set import BiozernikeDataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

learningRate = 1e-5
epochs = 1000
batch_size = 2 ** 10
l2_weight = 1
evaluation_step = 1000

cath_coefficients_file = sys.argv[0]
ecod_coefficients_file = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

dataset = BiozernikeDataset(cath_coefficients_file)
weights = dataset.weights()
sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=4)

testing_set = BiozernikeDataset(ecod_coefficients_file)
test_dataloader = DataLoader(testing_set, batch_size=len(testing_set))
x_test, y_test = next(iter(test_dataloader))

writer = SummaryWriter()

model = LogisticRegressionUniform(input_features=3922).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

writer.add_graph(model, x_test[0])

batch_n = 0
print("Starting training. Number of batches: %s" % (len(train_dataloader)))
for epoch in range(epochs):
    loss = None
    start = time.time()
    for x_train, y_train in train_dataloader:
        optimizer.zero_grad()
        y_predicted = model(x_train.to(device))
        loss = criterion(y_predicted, y_train.to(device)) + l2_weight * torch.sum(model.get_weights() ** 2)
        loss.backward()
        optimizer.step()

        if batch_n % evaluation_step == 0:
            with torch.no_grad():
                end = time.time()
                y_evaluation = model(x_test.to(device)).to("cpu")
                roc_auc = roc_auc_score(y_test.detach().numpy(), y_evaluation.detach().numpy())
                precision, recall, thresholds = precision_recall_curve(
                    y_test.detach().numpy(),
                    y_evaluation.detach().numpy()
                )
                pr_auc = auc(recall, precision)
                print("epoch %s, batch number %s, loss %s, testing roc %s auc %s. Time %d" % (
                    epoch,
                    batch_n,
                    loss.item(),
                    roc_auc,
                    pr_auc,
                    (end - start)*1000
                ))
                writer.add_scalar("Loss/1000batches", loss, batch_n//1000)
                writer.add_scalar("ROC/1000batches", roc_auc, batch_n//1000)
                writer.add_scalar("AUC/1000batches", pr_auc, batch_n//1000)
                writer.add_histogram("Weights/1000batches", model.get_weights(), batch_n//1000)
                writer.add_histogram("Bias/1000batches", model.get_bias(), batch_n//1000)
                sorted_evaluation = torch.Tensor.tolist(torch.squeeze(torch.argsort(y_evaluation, dim=0, descending=True)))
                writer.add_text(
                    "High Prediction/Last batch",
                    "  \n".join([
                        "%s : %s : %s" % (testing_set.get_classes(idx), y_test[idx, 0].item(), y_evaluation[idx, 0].item())
                        for idx in sorted_evaluation[0:1000]
                    ])
                )
                writer.add_text(
                    "Low Prediction/Last batch",
                    "  \n".join([
                        "%s : %s : %s" % (testing_set.get_classes(idx), y_test[idx, 0].item(), y_evaluation[idx, 0].item())
                        for idx in sorted_evaluation[len(sorted_evaluation)-1000:len(sorted_evaluation)]
                    ])
                )
                writer.flush()
                start = time.time()

        batch_n += 1

writer.close()
