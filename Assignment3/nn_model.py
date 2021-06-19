import json
import pprint
from typing import Tuple

import pandas
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
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        data = self.lin1(data)
        data = torch.relu(data)
        data = self.lin_hid(data)
        data = self.dropout(data)
        data = torch.tanh(data)
        data = self.lin2(data)
        return self.softmax(data)

def create_loader(train_data, train_pred, batch_size):
    trainset = Dataset(train_data, train_pred)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    return trainloader


def prepare_dataset(input_data, output_data, device, batch_size, testing_percentage=0.15, validation_percentage=0.15):
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


def network_training_testing(input_data, output_data, input_size, output_size, device, batch_size):
    train_loader, validation_loader, testing_loader = prepare_dataset(input_data, output_data, batch_size)
    nn = MLP(input_size=input_size, hidden_size=5, output_size=output_size).to(device)
    nn, training_losses, validation_losses = train_network(nn, train_loader, validation_loader)
    metric = test_network(testing_loader, nn)

    with open("results.json", "w")as fp:
        json.dump({
        "trainin_loss": training_losses,
        "validation_loss": validation_losses,
        "metrics": metric,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }, fp)

    return nn


def network_validation(network, criterion, validation_dataset, device="cuda") -> Tuple[list, list]:
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

        tot += labels.size(0)
        corr += (predicted == labels.cpu()).sum().item()

        epoch_validation_accuracy.append(corr/tot)
        epoch_validation_loss.append(loss.item())

    return epoch_validation_loss, epoch_validation_accuracy


def train_network(
    network: MLP, trainloader: DataLoader, validation_loader: DataLoader, learning_rate, epochs, device) -> Tuple[MLP, list, list]:
    total_iterations = 0
    total_losses = []
    validation_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

     # optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(epochs):  # loop over the dataset multiple times
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
        print('Training...')
        running_loss = 0.0
        network.train()

        for batch_idx, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.reshape([inputs.shape[0], -1]).to(torch.float32).to(device)
            optimizer.zero_grad()
            outputs = network(inputs.to(device))
            loss = criterion(outputs.to(device), labels.to(device))

            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

            loss.backward()
            optimizer.step()

            total_iterations += 1
            total_losses.append(loss.item())
            running_loss += loss.item()
            if batch_idx % 2000 == 1999:
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, batch_idx + 1, running_loss / 2000)
                )
                running_loss = 0.0
        validation_losses += network_validation(network, criterion, validation_loader, device)

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
            tot += labels.size(0)
            corr += (predicted == labels).sum().item()

            predictions += predicted
            true_labels += labels.to("cpu")

    acc = 100 * corr / tot
    print("Accuracy of the network on the %d test images: %d %%" % (counter, acc))

    res = metrics(predictions, true_labels)
    res["test_accuracy"] = acc

    return res


def metrics(pred_flat, labels_flat) -> dict:
    """Function to various metrics of our predictions vs labels"""
    print(json.dumps(classification_report(labels_flat, pred_flat, output_dict=True)))
    print("\n**** Classification report")
    print(classification_report(labels_flat, pred_flat))

    print("\n***Confusion matrix")
    pprint.pprint(confusion_matrix(pred_flat, labels_flat))

    plot_class_report = pandas.DataFrame(classification_report(pred_flat, labels_flat))
    fig = plot_class_report.plot(kind='bar', x="dataframe_1", y="dataframe_2")  # bar can be replaced by
    fig.savefig("classif_report.png", dpi=200, format='png', bbox_inches='tight')
    plt.close()

    labels = ["class 1", "class 2", "class 3"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix(pred_flat, labels_flat))
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("conf_matrix.png", dpi=200, format='png', bbox_inches='tight')

    info = {
        "classification_report": json.dumps(classification_report(labels_flat, pred_flat)),
        # "confusion_matrix": json.dumps(confusion_matrix(pred_flat, labels_flat))
    }
    return info