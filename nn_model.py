import torch


class TitanicPredictor(torch.nn.Module):
    def __init__(self, num_features, num_targets, pclass_encoder):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features=num_features, out_features=num_features)
        self.out = torch.nn.Linear(in_features=num_features, out_features=num_targets)
        self.pclass_encoder = pclass_encoder

    def forward(self, x):
        real_features = x[:, 1:]
        encoded_features = torch.nn.functional.one_hot(x[:, 0].long() - 1)
        sig = torch.nn.Sigmoid()
        pclass_out = self.pclass_encoder.forward(encoded_features)
        x = torch.cat((pclass_out, real_features), 1)
        x = self.fc1(x)
        x = sig(self.out(x))
        return x


class Encoder(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = torch.nn.Linear(in_features=num_classes, out_features=1)

    def forward(self, x):
        return self.fc(x.float())


def train_model(orig_features, labels):
    features = orig_features.clone()
    pclass_encoder = Encoder(num_classes=3)
    model = TitanicPredictor(len(features[0]), 1, pclass_encoder)
    loss_fn = torch.nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimzer, step_size=100, gamma=.8)

    EPOCHS = 4000
    loss_list = []

    for i in range(EPOCHS):
        epoch_num = i + 1
        predictions = model.forward(features)
        loss = loss_fn(predictions, labels)
        loss_list.append(loss)

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        scheduler.step()
        if epoch_num % 100 == 0:
            print(f'Epoch: {epoch_num}, Loss: {loss}, lr: {scheduler.state_dict()["_last_lr"][0]}')

    return model

