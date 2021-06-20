import json
import pprint
from typing import Tuple

import numpy
import pandas
import pandas as pd
import seaborn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, inps, labels):
        """Initialization"""
        self.labels = labels
        self.inps = inps

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.inps)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        x = self.inps[index]
        y = int(self.labels[index])

        return x, y


class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin_hid = torch.nn.Linear(hidden_size, int(hidden_size/2))
        self.lin2 = torch.nn.Linear(int(hidden_size/2), output_size)
        self.softmax = torch.nn.Softmax(1)
        # self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.sigmoid(data)
        data = self.lin_hid(data)
        # data = self.dropout(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)

def create_loader(train_data, train_pred, batch_size):
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_dataset(input_data, output_data, device, batch_size, testing_percentage=0.15, validation_percentage=0.15):
    input_data = input_data.to_numpy()
    output_data = output_data.to_numpy()
    train_data = input_data[int((validation_percentage + testing_percentage) * len(input_data)):]
    train_labels = output_data[int((validation_percentage + testing_percentage) * len(output_data)):]

    train_data = torch.tensor(train_data, device=device)
    train_labels= torch.tensor(train_labels, device=device)
    train_loader = create_loader(train_data, train_labels, batch_size)
    print("divided train data")

    validation_data = input_data[:int(validation_percentage * len(input_data))]
    validation_labels = output_data[: int(validation_percentage * len(output_data))]

    validation_data = torch.tensor(validation_data, device=device)
    validation_labels= torch.tensor(validation_labels, device=device)
    validation_loader = create_loader(validation_data, validation_labels, batch_size)
    print("divided validation data")

    testing_data = input_data[int(validation_percentage * len(input_data)): int(validation_percentage * len(input_data)) + int(
        testing_percentage * len(input_data))]
    testing_labels = output_data[int(validation_percentage * len(output_data)): int(validation_percentage * len(output_data)) + int(
        testing_percentage * len(output_data))]

    testing_data = torch.tensor(testing_data, device=device)
    testing_labels = torch.tensor(testing_labels, device=device)
    testing_loader = create_loader(testing_data, testing_labels, batch_size)
    print("divided test data")

    return train_loader, validation_loader, testing_loader


def network_training_testing(input_data, output_data, input_size, output_size, device, batch_size, learning_rate, epochs):
    train_loader, validation_loader, testing_loader = prepare_dataset(input_data, output_data, device, batch_size)
    print(f"Train loader: {len(train_loader)}")
    print(f"Validation loader: {len(validation_loader)}")
    print(f"Testing loader: {len(testing_loader)}")
    nn = MLP(input_size=input_size, hidden_size=100, output_size=output_size).to(device)
    nn, training_losses, validation_losses = train_network(nn, train_loader, validation_loader, learning_rate, epochs, device)
    metric = test_network(testing_loader, nn, device)

    with open("results.json", "w")as fp:
        json.dump({
        "trainin_loss": training_losses,
        "validation_loss": validation_losses,
        "metrics": metric,
        "learning_rate": learning_rate,
    }, fp)

    return nn


def network_validation(network, criterion, validation_dataset, device="cuda") -> Tuple[list, float]:
    """Validation phase, executed after each epoch of the training phase to see the improvements of the network."""
    epoch_validation_loss = []
    epoch_validation_accuracy = []
    corr = 0
    tot = 0
    for batch_idx, (inputs, labels) in tqdm(enumerate(validation_dataset)):
        inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

        with torch.no_grad():
            outs = network(inputs.to(device))
        loss = criterion(outs, labels.to(device))
        _, predicted = torch.max(outs.data, 1)
        # print(predicted)
        # print(labels.cpu())
        tot += labels.size(0)
        corr += (predicted.cpu() == labels.cpu()).sum().item()

        epoch_validation_accuracy.append(corr/tot)
        epoch_validation_loss.append(loss.item())

    return epoch_validation_loss, sum(epoch_validation_accuracy)/len(epoch_validation_accuracy)


def train_network(
    network: MLP, trainloader: DataLoader, validation_loader: DataLoader, learning_rate, epochs, device) -> Tuple[MLP, list, list]:
    total_iterations = 0
    total_losses = []
    validation_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        running_loss = 0.0
        network.train()
        epoch_losses = []
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = network(inputs.to(device))
            loss = criterion(outputs.to(device), labels.to(device))

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            total_iterations += 1
            epoch_losses.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 2000)
                )
                running_loss = 0.0
        validation_losses.append(network_validation(network, criterion, validation_loader, device))
        total_losses.append(epoch_losses)
    temp_losses = [sum(ep_loss)/len(ep_loss) for ep_loss in total_losses]
    print(f"Train loss for the MLP classifier: {temp_losses}")
    plt.plot(temp_losses)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("MLP Training Loss")
    plt.show()

    val_losses = numpy.array([sum(val[0])/len(val[0]) for val in validation_losses])
    # val_losses = numpy.average(val_losses, axis=0)

    val_accuracy = [val[1] for val in validation_losses]
    # val_accuracy = numpy.average(val_accuracy, axis=0).tolist()

    plt.plot(val_losses, label="Validation average loss")
    plt.plot(val_accuracy, label="Validation average accuracy")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("MLP Validation Loss & Accuracy ")
    plt.legend()
    plt.show()
    return network, total_losses, validation_losses


def test_network(testloader: DataLoader, network: MLP, device) -> dict:
    corr = 0
    tot = 0
    counter = 0
    predictions, true_labels = [], []
    network.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)

            counter += 1
            outputs = network(inputs.to(device))
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.cpu()
            tot += labels.size(0)
            corr += (predicted == labels.cpu()).sum().item()

            predictions += predicted
            true_labels += labels.to("cpu")

    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test set: %d %%" % (counter, acc))

    res = metrics(predictions, true_labels)
    res["test_accuracy"] = acc

    return res


def metrics(labels_flat, pred_flat, name: str="MLP") -> dict:
    """Function to various metrics of our predictions vs labels"""
    print(json.dumps(classification_report(labels_flat, pred_flat, output_dict=True)))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))

    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    plot_class_report = pd.DataFrame.from_dict(classification_report(pred_flat, labels_flat, output_dict=True))
    print(plot_class_report)
    print(plot_class_report.columns)
    # fig = plot_class_report.plot(kind='bar', x="dataframe_1", y="dataframe_2")  # bar can be replaced by
    fig = plot_class_report.plot.bar(rot=1)
    fig.figure.savefig(name + "_classif_report.png", dpi=200, format='png', bbox_inches='tight')
    plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    conf_matrix = pd.DataFrame(confusion_matrix(pred_flat, labels_flat), index=[0,1,2,3], columns=[0,1,2,3])
    plt.figure(figsize=(10,7))
    seaborn.heatmap(conf_matrix, annot=True)
    # cax = ax.matshow(conf_matrix, annot=True)
    # plt.title('Confusion matrix of the classifier')
    # fig.colorbar(cax)
    # ax.set_xticklabels(unique_labels)
    # ax.set_yticklabels(unique_labels)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    plt.savefig(name + "_conf_matrix.png", dpi=200, format='png', bbox_inches='tight')

    info = {
        "classification_report": classification_report(labels_flat, pred_flat, output_dict=True),
        # "confusion_matrix": json.dumps(confusion_matrix(pred_flat, labels_flat))
    }
    return info
