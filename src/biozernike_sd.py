import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.fully_connected_nn import FullyConnectedSigmoid, FullySigmoid, LogisticRegression
from src.biozernike_data_set import BiozernikeDataset

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

if __name__ == '__main__':
    learning_rate = 1e-6
    epochs = 10000
    batch_size = 2 ** 8
    l2_weight = 1e0
    evaluation_step = 10000
    hidden_layer = 2 ** 10

    cath_coefficients_file = "../resources/cath_moments.tsv"
    ecod_coefficients_file = "../resources/ecod_moments.tsv"

    dataset = BiozernikeDataset(cath_coefficients_file)
    weights = dataset.weights()
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    testing_set = BiozernikeDataset(ecod_coefficients_file)
    test_dataloader = DataLoader(testing_set, batch_size=len(testing_set))
    x_test, y_test = next(iter(test_dataloader))

    writer = SummaryWriter()

    model = FullyConnectedSigmoid(input_features=3922, hidden_layer=hidden_layer)
    # model = LogisticRegression(input_features=3922)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.get_params(), lr=learning_rate, weight_decay=l2_weight)

    writer.add_graph(model, x_test[0])
    writer.add_text(
        "Description",
        "learning-rate: %s  \nbatch-size: %s  \nl2-weight: %s  \nhidden-layer: %s  \nmodel-name: %s  \noptimizer: %s  \nloss: %s  \n"
        % (learning_rate, batch_size, l2_weight, hidden_layer, type(model).__name__, type(optimizer).__name__, type(criterion).__name__)
    )

    print("Starting training. Number of batches: %s" % (len(train_dataloader)))
    loss = None
    for epoch in range(epochs):
        for x_train, y_train in train_dataloader:
            optimizer.zero_grad()
            y_predicted = model(x_train)
            loss = criterion(y_predicted, y_train)
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
        sorted_evaluation = torch.Tensor.tolist(
            torch.squeeze(torch.argsort(y_evaluation, dim=0, descending=True))
        )

        for name, param in model.get_weights():
            writer.add_histogram("%s/epoch" % name, param, epoch)

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
