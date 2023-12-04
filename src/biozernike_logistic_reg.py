import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.networks.logistic_regr_nn import LogisticRegression
from src.data_set.biozernike_data_set import BiozernikeDataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

if __name__ == '__main__':
    learning_rate = 1e-6
    epochs = 10000
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

    writer = SummaryWriter()

    model = LogisticRegression(input_features=3922)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    writer.add_graph(model, x_test[0])
    writer.add_text("learning-rate", "%s" % learning_rate)
    writer.add_text("batch-size", "%s" % batch_size)
    writer.add_text("l2-weight", "%s" % l2_weight)
    writer.add_text("model-name", "%s" % type(model).__name__)

    print("Starting training. Number of batches: %s" % (len(train_dataloader)))
    loss = None
    for epoch in range(epochs):
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train) + l2_weight * torch.sum(model.get_weights() ** 2)
            loss.backward()
            optimizer.step()

        y_evaluation = model(x_test)
        roc_auc = roc_auc_score(y_test.detach().numpy(), y_evaluation.detach().numpy())
        precision, recall, thresholds = precision_recall_curve(
            y_test.detach().numpy(),
            y_evaluation.detach().numpy()
        )
        pr_auc = auc(recall, precision)
        print("epoch %s, loss %s, testing roc %s auc %s" % (
            epoch,
            loss.item(),
            roc_auc,
            pr_auc
        ))
        writer.add_scalar("Loss/epoch", loss, epoch)
        writer.add_scalar("ROC/epoch", roc_auc, epoch)
        writer.add_scalar("AUC/epoch", pr_auc, epoch)
        writer.add_histogram("Weights/epoch", model.get_weights(), epoch)
        writer.add_histogram("Bias/epoch", model.get_bias(), epoch)
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

    writer.close()
