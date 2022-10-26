import torch
from datetime import datetime


class TrainableNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(TrainableNet, self).__init__()

    def fit(self, X, Y, epochs=50, lr=0.001):
        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        epochs = epochs

        self.train()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print("Using device:", device)
        if device != "cpu":
            self.cuda()

        print("Dataset size ", len(X))
        batch_size = 32
        batches = []
        for i in range(0, len(X), batch_size):
            batch = (X[i : i + batch_size], Y[i : i + batch_size])
            tensor_batch = (
                torch.tensor(batch[0], dtype=torch.float32),
                torch.tensor(batch[1], dtype=torch.float32),
            )
            if device != "cpu":
                tensor_batch = (tensor_batch[0].cuda(), tensor_batch[1].cuda())
            batches.append(tensor_batch)

        target_loss = 0.1
        iters_per_epoch = 10
        for epoch in range(epochs):
            t0 = datetime.now()
            running_loss = 0.0
            for iter in range(iters_per_epoch):
                for batch in batches:
                    x, y = batch

                    # reshape y
                    y = y.view(-1, 1)

                    # print(x.shape, y.shape)
                    y_pred = self.forward(x)
                    # print(y_pred.shape)

                    loss = criterion(y_pred, y)

                    running_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            t1 = datetime.now()
            elapsed_time_in_seconds = (t1 - t0).total_seconds()
            print(
                f"Epoch {epoch} loss {running_loss / batch_size / iters_per_epoch} (used time: {elapsed_time_in_seconds} seconds)"
            )
            if running_loss / batch_size / iters_per_epoch < target_loss:
                print("Target loss reached, early stopping")
                break

    def train_with_eval(self, X, Y, X_eval, Y_eval, epochs=50, batch_size=32, lr=0.001):
        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        epochs = epochs

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = "cpu"
        print("Using device:", device)
        if device != "cpu":
            self.cuda()

        print("Dataset size ", len(X))
        batches = []
        for i in range(0, len(X), batch_size):
            batch = (X[i : i + batch_size], Y[i : i + batch_size])
            tensor_batch = (
                torch.tensor(batch[0], dtype=torch.float32),
                torch.tensor(batch[1], dtype=torch.float32),
            )
            if device != "cpu":
                tensor_batch = (tensor_batch[0].cuda(), tensor_batch[1].cuda())
            batches.append(tensor_batch)

        target_loss = 0.1
        iters_per_epoch = 1
        min_avg_loss = 100.0
        best_state = None

        for epoch in range(epochs):
            self.train()
            t0 = datetime.now()
            running_loss = 0.0
            for iter in range(iters_per_epoch):
                for batch in batches:
                    x, y = batch

                    # reshape y
                    y = y.view(-1, 1)

                    # print(x.shape, y.shape)
                    y_pred = self.forward(x)
                    # print(y_pred.shape)

                    loss = criterion(y_pred, y)

                    running_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            t1 = datetime.now()
            elapsed_time_in_seconds = (t1 - t0).total_seconds()
            eval_prediction = self.predict(X_eval)
            loss = criterion(torch.tensor(eval_prediction), torch.tensor(Y_eval))
            eval_loss = loss.item()
            training_loss = running_loss / batch_size / iters_per_epoch
            # Average loss applies more weight to evaluation loss
            average_loss = (training_loss + eval_loss * 3) / 4
            print(
                f"Epoch {epoch} loss {running_loss / batch_size / iters_per_epoch} (used time: {elapsed_time_in_seconds} seconds) || Evaluation loss {eval_loss} || Average loss {average_loss}"
            )
            if average_loss < min_avg_loss:
                min_avg_loss = average_loss

                # print("Saving model")
                best_state = self.state_dict()

            if average_loss < target_loss:
                print("Target loss reached, early stopping")
                break

        print("Saving best state...")
        torch.save(best_state, "model.pt")

        print("Best average loss", min_avg_loss)
        print("Loading best model with average loss")
        self.load_state_dict(torch.load("model.pt"))

    def predict(self, X):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device != "cpu":
            self.cuda()
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            if device != "cpu":
                x = x.cuda()
            # print(x.device, self.fc.weight.device)
            y_pred = self.forward(x)
            # print(y_pred)
            return y_pred.detach().cpu().numpy()


class LinearNet(TrainableNet):
    def __init__(self, in_features, hidden_size=1024, dropout=0.2):
        super(TrainableNet, self).__init__()
        self.hidden_size = hidden_size

        self.bn1 = torch.nn.BatchNorm1d(in_features)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(in_features, self.hidden_size)

        self.relu = torch.nn.LeakyReLU()

        self.dropout2 = torch.nn.Dropout(dropout)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_size)

        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.relu2 = torch.nn.LeakyReLU()

    def forward(self, x):
        y = self.bn1(x)
        y = self.dropout1(y)
        y = self.fc(y)
        y = self.relu(y)
        y = self.dropout2(y)
        y = self.bn2(y)
        y = self.fc2(y)
        y = self.relu2(y)
        return y


def test_linear_net():
    net = LinearNet(5)
    result = net.forward(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    print(result)


if __name__ == "__main__":
    net = LinearNet(5)

    X = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [2.0, 2.0, 3.0, 4.0, 5.0],
        [3.0, 3.0, 3.0, 4.0, 5.0],
        [4.0, 4.0, 4.0, 4.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
    ]
    y = [1.0, 2.0, 3.0, 4.0, 5.0]

    net.fit(X, y)

    pred = net.predict(X)
    print(pred)
