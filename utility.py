import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
import nn_model
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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


def get_test_data():
    return pd.read_csv('data/test.csv', converters=TITANIC_DATA_CONVERTERS)


def get_training_data(convert=False):
    if convert:
        return pd.read_csv('data/train.csv', converters=TITANIC_DATA_CONVERTERS)
    else:
        return pd.read_csv('data/train.csv')


def print_metrics(predictions, labels, model_name):
    performances = confusion_matrix(y_true=labels, y_pred=predictions)
    precision = performances[1][1] / (performances[1][1] + performances[0][1])
    recall = performances[1][1] / (performances[1][1] + performances[1][0])
    accuracy = (performances[1][1] + performances[0][0]) / len(labels)

    print(f'{model_name}\n\tPrecision: {precision}, Recall: {recall}, Accuracy: {accuracy}')


def train_model(features, labels):
    model = nn_model.TitanicPredictor(len(features[0]), 1)
    loss_fn = torch.nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimzer, lr_lambda=lambda epoch: .98)

    EPOCHS = 30000
    loss_list = []

    for i in range(EPOCHS):
        epoch_num = i + 1
        predictions = model.forward(features)
        loss = loss_fn(predictions, labels)
        loss_list.append(loss)

        if epoch_num % 100 == 0:
            print(f'Epoch: {epoch_num} Loss: {loss}')

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()
        if epoch_num % 100 == 0:
            scheduler.step()
            if epoch_num % 500 == 0:
                print(f'\tlr is {scheduler.state_dict()["_last_lr"]}')

    return model

# df = get_training_data()
# create_two_feature_plot(df, 'Sex', 'Age', feat1_range=['male', 'female'])
# create_two_feature_plot(df, 'Sex', 'Pclass', feat1_range=['male', 'female'], add_y_jitter=True)


train_df = get_training_data(convert=True)
features_for_training = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
train_features = torch.as_tensor(train_df[features_for_training].values.astype(np.float), dtype=torch.float)
train_labels = torch.as_tensor(train_df['Survived'].values.astype(np.float), dtype=torch.float).unsqueeze(1).detach()

nn_model = train_model(train_features, train_labels)
training_predictions = nn_model.forward(train_features.detach()).detach().round()
print_metrics(predictions=training_predictions, labels=train_labels, model_name='Neural Network')

forest_model = RandomForestClassifier(max_depth=None, n_estimators=500, random_state=1)
forest_model.fit(train_df[features_for_training].values, train_df['Survived'].values)
training_predictions = forest_model.predict(train_df[features_for_training].values)
print_metrics(predictions=training_predictions, labels=train_labels, model_name='Random Forest')

# Generate predictions to submit
# test_df = get_test_data()
# test_features = torch.as_tensor(test_df[features_for_training].values.astype(np.float), dtype=torch.float)
# predictions = model.forward(test_features)


