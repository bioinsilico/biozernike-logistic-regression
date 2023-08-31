import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.logistic_regr_nn import LogisticRegression, LogisticRegressionUniform
from src.biozernike_data_set import BiozernikeDataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

if __name__ == '__main__':
    learning_rate = 1e-6
    epochs = 2000
    batch_size = 2 ** 8
    l2_weight = 1
    evaluation_step = 1000

    cath_coefficients_file = "../resources/cath_moments.tsv"
    ecod_coefficients_file = "../resources/ecod_moments.tsv"

    dataset = BiozernikeDataset(cath_coefficients_file)
    weights = dataset.weights()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=2)

    testing_set = BiozernikeDataset(ecod_coefficients_file)
    test_dataloader = DataLoader(testing_set, batch_size=len(testing_set))
    x_test, y_test = next(iter(test_dataloader))

    writer = SummaryWriter(comment=" | learning-rate: %s batch-size: %s l2-weight: %s" % (
        learning_rate,
        batch_size,
        l2_weight
    ))

    model = LogisticRegression(input_features=3922)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer.add_graph(model, x_test[0])

    batch_n = 0
    print("Starting training. Number of batches: %s" % (len(train_dataloader)))
    for epoch in range(epochs):
        loss = None
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train) + l2_weight * torch.sum(model.get_weights() ** 2)
            loss.backward()
            optimizer.step()

            if batch_n % evaluation_step == 0:
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
                writer.add_scalar("Loss/1000batches", loss, batch_n // 1000)
                writer.add_scalar("ROC/1000batches", roc_auc, batch_n // 1000)
                writer.add_scalar("AUC/1000batches", pr_auc, batch_n // 1000)
                writer.add_histogram("Weights/1000batches", model.get_weights(), batch_n // 1000)
                writer.add_histogram("Bias/1000batches", model.get_bias(), batch_n // 1000)
                sorted_evaluation = torch.Tensor.tolist(
                    torch.squeeze(torch.argsort(y_evaluation, dim=0, descending=True))
                )
                writer.add_text(
                    "High Prediction/Last batch",
                    "  \n".join([
                        "%s : %s : %s" % (
                            testing_set.get_classes(idx), y_test[idx, 0].item(), y_evaluation[idx, 0].item())
                        for idx in sorted_evaluation[0:1000]
                    ])
                )
                writer.add_text(
                    "Low Prediction/Last batch",
                    "  \n".join([
                        "%s : %s : %s" % (
                            testing_set.get_classes(idx), y_test[idx, 0].item(), y_evaluation[idx, 0].item())
                        for idx in sorted_evaluation[len(sorted_evaluation) - 1000:len(sorted_evaluation)]
                    ])
                )
                writer.flush()

            batch_n += 1

    writer.close()
