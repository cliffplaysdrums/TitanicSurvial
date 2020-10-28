import torch


class TitanicPredictor(torch.nn.Module):
    def __init__(self, num_features, num_targets):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=num_features)
        self.out = torch.nn.Linear(in_features=num_features, out_features=num_targets)

    def forward(self, x):
        sig = torch.nn.Sigmoid()
        x = self.fc1(x)
        x = sig(self.out(x))
        return x


def train_model(features, labels):
    model = TitanicPredictor(len(features[0]), 1)
    loss_fn = torch.nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimzer, lr_lambda=lambda epoch: .9)

    EPOCHS = 2000
    loss_list = []

    for i in range(EPOCHS):
        epoch_num = i + 1
        predictions = model.forward(features)
        loss = loss_fn(predictions, labels)
        loss_list.append(loss)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if epoch_num % 50 == 0:
            scheduler.step()
            if epoch_num % 100 == 0:
                print(f'Epoch: {epoch_num}, Loss: {loss}, lr: {scheduler.state_dict()["_last_lr"][0]}')

    return model
