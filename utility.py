import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
import nn_model
import torch

np.random.seed(1)


def convert_sex(sex):
    if sex == 'male':
        return 0
    elif sex == 'female':
        return 1
    else:
        raise ValueError(f'Expected "male" or "female" but got {sex}.')


# It's possible cabin has some relevance, but we need to look more into how they're specified because sometimes we're
# given multiple rooms and even multiple sections. E.g. 'F23' or 'F G73' or 'F23 F25 F27', etc
# def convert_cabin(cabin):
#     print(f'Got cabin {cabin} of type {type(cabin)}')
#     if cabin is None or len(cabin) == 0:
#         return 0
#     else:
#         section = (ord(cabin[0]) - ord('A')) * 1000
#         print(section)
#         num = 0
#         if len(cabin) > 1:
#             num = int(cabin[1:].split()[0])
#
#         return section + num


def convert_embarked(emb):
    if len(emb) == 0:
        return 0
    else:
        return ord(emb) - ord('B')


# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
TITANIC_DATA_CONVERTERS = {
    4: convert_sex,
    # 10: convert_cabin,
    11: convert_embarked
}


def create_two_feature_plot(df, feat1, feat2, feat1_range=None, add_y_jitter=False):
    output_file(f'plots/{feat1}-{feat2}.html')

    survivor_indices = df['Survived'] == 1
    x_survivors = df.loc[survivor_indices, feat1].values
    x_casualties = df.loc[~survivor_indices, feat1].values
    y_survivors = df.loc[survivor_indices, feat2].values
    y_casualties = df.loc[~survivor_indices, feat2].values

    # Add some jitter so we don't have too many points overlapping and can see them all
    dev = .05
    if feat1_range is not None:
        x_survivors = list(zip(x_survivors, np.random.normal(scale=dev, size=len(x_survivors))))
        x_casualties = list(zip(x_casualties, np.random.normal(scale=dev, size=len(x_casualties))))

    if add_y_jitter:
        y_survivors = y_survivors.astype(np.float) + np.random.normal(scale=dev, size=y_survivors.shape)
        y_casualties = y_casualties.astype(np.float) + np.random.normal(scale=dev, size=y_casualties.shape)

    p = figure(title=f'Titanic Passenger Survival', x_range=feat1_range, x_axis_label=feat1, y_axis_label=feat2)
    p.scatter(x_survivors, y_survivors, size=4, color='#3A5785', legend_label='Survivors')
    p.scatter(x_casualties, y_casualties, size=4, alpha=.5, color='#B5A87A', legend_label='Casualties')
    show(p)


def get_training_data(convert=False):
    if convert:
        return pd.read_csv('data/train.csv', converters=TITANIC_DATA_CONVERTERS)
    else:
        return pd.read_csv('data/train.csv')


def train_model(features, labels):
    model = nn_model.TitanicPredictor(len(features[0]), 1)
    loss_fn = torch.nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 1000
    loss_list = []

    for i in range(EPOCHS):
        epoch_num = i + 1
        predictions = model.forward(features)
        loss = loss_fn(predictions, labels)
        loss_list.append(loss)

        if epoch_num % 25 == 0:
            print(f'Epoch: {epoch_num} Loss: {loss}')

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

# df = get_training_data()
# create_two_feature_plot(df, 'Sex', 'Age', feat1_range=['male', 'female'])
# create_two_feature_plot(df, 'Sex', 'Pclass', feat1_range=['male', 'female'], add_y_jitter=True)


df = get_training_data(convert=True)
features_for_training = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
train_features = torch.as_tensor(df[features_for_training].values.astype(np.float), dtype=torch.float)
train_labels = torch.as_tensor(df['Survived'].values.astype(np.float), dtype=torch.float).unsqueeze(1)
train_model(train_features, train_labels)


