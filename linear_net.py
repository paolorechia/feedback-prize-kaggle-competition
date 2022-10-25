import torch


class LinearNet(torch.nn.Module):
    def __init__(self, in_features):
        super(LinearNet, self).__init__()
        self.hidden_size = 128
        self.fc = torch.nn.Linear(in_features, 1)
        # self.relu = torch.nn.ReLU()
        # self.dropout = torch.nn.Dropout(0.05)
        # self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        # self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        y = self.fc(x)
        # y = self.relu(y)
        # y = self.dropout(y)
        # y = self.fc2(y)
        # y = self.relu2(y)
        return y

    def fit(self, X, Y, epochs=100, lr=0.0001):
        criterion = torch.nn.MSELoss(reduction="mean")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        epochs = epochs

        self.train()

        print("Dataset size ", len(X))
        batch_size = 32

        target_loss = 0.0001
        iters_per_epoch = 20
        for epoch in range(epochs):
            running_loss = 0.0
            for iter in range(iters_per_epoch):
                for i in range(0, len(X), batch_size):

                    x = X[i : i + batch_size]
                    y = Y[i : i + batch_size]

                    x = torch.tensor(x, dtype=torch.float32)
                    y = torch.tensor(y, dtype=torch.float32)

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
            print(f"Epoch {epoch} loss {running_loss / batch_size / iters_per_epoch}")
            if running_loss / batch_size / iters_per_epoch < target_loss:
                print("Target loss reached, early stopping")
                break

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            x.cuda()
            y_pred = self.forward(x)
            # print(y_pred)
            return y_pred.detach().cpu().numpy()


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
