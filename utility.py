import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
import nn_model
from nn_model import train_model
from forest_model import search_forest_params
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler

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

TITANIC_TEST_DATA_CONVERTERS = {
    3: convert_sex,
    # 10: convert_cabin,
    10: convert_embarked
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


def engineer_features(features, num_new_features, algorithm='ica'):
    """Performs component analysis of the provided features.

    Args:
        features (np.ndarray): Features on which to perform component analysis.
        num_new_features (int): The number of components to which the features space will be reduced.
        algorithm (str): One of 'ica' or 'pca'

    Returns:
        (tuple): A numpy array of the original features concatenated column-wise with the features projected into the
        component space generated by the component analysis, and the algorithm model fitted to the data to allow future
        projections of new data.
    """
    projector = None
    if algorithm == 'ica':
        projector = FastICA(n_components=num_new_features, random_state=1)
    elif algorithm == 'pca':
        projector = PCA(n_components=num_new_features, random_state=1)

    new_components = projector.fit_transform(features)
    return np.concatenate([features, new_components], axis=1), projector


def get_test_data(normalize=None):
    """Loads Titanic challenge test data.

    Args:
        normalize (list): List of column names to scale proportionally to the inclusive range (-1, 1).

    Returns:
        df (pd.DataFrame): The test data.
    """
    df = pd.read_csv('data/test.csv', converters=TITANIC_TEST_DATA_CONVERTERS).fillna(value=0)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    if normalize is not None:
        for col in normalize:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


def get_training_data(convert=False, normalize=None):
    """Loads Titanic challenge training data.

    Args:
        convert (bool): Convert categorical fields to int.
        normalize (list): List of column names to scale proportionally to the inclusive range (-1, 1).

    Returns:
        df (pd.DataFrame): The training data.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = None

    if convert:
        df = pd.read_csv('data/train.csv', converters=TITANIC_DATA_CONVERTERS)
    else:
        df = pd.read_csv('data/train.csv')

    if normalize is not None:
        for col in normalize:
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


def make_torch_predictions(model, features):
    """

    Args:
        model (nn_model.TitanicPredictor):
        features (np.ndarray):

    Returns:
        predictions (np.ndarray)
    """
    with torch.no_grad():
        test_features = torch.as_tensor(features, dtype=torch.float)
        predictions = model.forward(test_features).round().type(dtype=torch.int)
        return predictions.numpy()


def plot_projection(projected_features, labels, filename, plot_title):
    """Creates a plot of a 2D array, color-coded according to labels. Intended to plot results of components analysis.

    Args:
        projected_features (np.ndarray): The values to be plotted. Has shape (n x 2).
        labels (np.ndarray): The label of each sample. Must have shape (n,) or (n,1).
        filename (str): The name to use when saving the file in the local 'plots' directory.
        plot_title (str): The title to display on the plot.

    Returns:
        None
    """
    output_file(f'plots/{filename}.html')

    survivor_indices = labels == 1
    x_survivors = projected_features[survivor_indices, 0]
    x_casualties = projected_features[~survivor_indices, 0]
    y_survivors = projected_features[survivor_indices, 1]
    y_casualties = projected_features[~survivor_indices, 1]

    p = figure(title=plot_title)
    p.scatter(x_survivors, y_survivors, size=4, color='#3A5785', legend_label='Survivors')
    p.scatter(x_casualties, y_casualties, size=4, alpha=.5, color='#B5A87A', legend_label='Casualties')
    show(p)


def print_metrics(predictions, labels, model_name):
    performances = confusion_matrix(y_true=labels, y_pred=predictions)
    precision = performances[1][1] / (performances[1][1] + performances[0][1])
    recall = performances[1][1] / (performances[1][1] + performances[1][0])
    accuracy = (performances[1][1] + performances[0][0]) / len(labels)

    print(f'{model_name}\n\tPrecision: {precision}, Recall: {recall}, Accuracy: {accuracy}')


def generate_forest_submission(trained_model, relevant_features, to_normalize):
    test_df = get_test_data(normalize=to_normalize)
    test_features_array = test_df[relevant_features].values.astype(np.float)

    # Make predictions & save
    predictions = trained_model.predict(test_features_array)
    to_submit = pd.DataFrame(np.concatenate(
        [test_df['PassengerId'].values.reshape((-1, 1)).astype(int), predictions.reshape(-1, 1)], axis=1).astype(int),
        columns=['PassengerId', 'Survived'])
    to_submit.to_csv(path_or_buf='data/forest_submission.csv', index=False, header=True)


def generate_nn_submission(train_features_array, train_labels_array, relevant_features):
    # Feature engineering
    ica = FastICA(n_components=2, random_state=1)
    train_features = np.concatenate([train_features_array, ica.fit_transform(train_features_array)], axis=1)
    # Same for test data
    test_df = get_test_data()
    test_features_array = test_df[relevant_features].values.astype(np.float)
    test_features = np.concatenate([test_features_array, ica.fit_transform(test_features_array)], axis=1)

    # Convert to tensors & train
    train_features_tensor = torch.as_tensor(train_features, dtype=torch.float)
    train_labels_tensor = torch.as_tensor(train_labels_array, dtype=torch.float).unsqueeze(1).detach()
    nn_model = train_model(train_features_tensor, train_labels_tensor)

    # Make predictions & save
    predictions = make_torch_predictions(nn_model, test_features)
    to_submit = pd.DataFrame(np.concatenate([test_df['PassengerId'].values.reshape((-1, 1)), predictions], axis=1),
                             columns=['PassengerId', 'Survived'])
    to_submit.to_csv(path_or_buf='data/nn_submission.csv', index=False, header=True)


# df = get_training_data()
# create_two_feature_plot(df, 'Sex', 'Age', feat1_range=['male', 'female'])
# create_two_feature_plot(df, 'Sex', 'Pclass', feat1_range=['male', 'female'], add_y_jitter=True)

to_normalize = ['Age', 'SibSp', 'Fare']
train_df = get_training_data(convert=True, normalize=to_normalize)
features_for_training = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
train_features_array = train_df[features_for_training].values.astype(np.float)
train_labels_array = train_df['Survived'].values.astype(np.float)

train_features, cv_features, train_labels, cv_labels = train_test_split(train_features_array, train_labels_array,
                                                                        train_size=.75, random_state=1)

# pca = PCA(n_components=8, random_state=1)
# poly = PolynomialFeatures(degree=3)

# train_features = poly.fit_transform(train_features)
# cv_features = poly.fit_transform(cv_features)
# train_features = pca.fit_transform(train_features)
# cv_features = pca.fit_transform(cv_features)

# Uncomment to add features from ICA
# n_components = 2
# train_features_with_ica, projector = engineer_features(train_features, num_new_features=n_components, algorithm='ica')
# cv_features_with_ica = projector.transform(cv_features)
# # plot_projection(train_features_with_ica[:, -n_components:], train_labels, filename=f'ica-{n_components}-components',
# #                 plot_title='ICA of Titanic Passengers')
# train_features = train_features_with_ica
# cv_features = np.concatenate([cv_features, cv_features_with_ica], axis=1)

# Uncomment to add features from PCA
# n_components = 2
# train_features_with_pca = engineer_features(train_features, num_new_features=n_components, algorithm='pca')
# cv_features_with_pca = engineer_features(cv_features, num_new_features=n_components, algorithm='pca')
# plot_projection(train_features_with_pca[:, -n_components:], train_labels, filename=f'pca-{n_components}-components',
#                 plot_title='PCA of Titanic Passengers')
# train_features = train_features_with_pca
# cv_features = cv_features_with_pca

# Create torch tensors to be used with NN
train_features_tensor = torch.as_tensor(train_features, dtype=torch.float)
train_labels_tensor = torch.as_tensor(train_labels, dtype=torch.float).unsqueeze(1).detach()
cv_features_tensor = torch.as_tensor(cv_features, dtype=torch.float)
cv_labels_tensor = torch.as_tensor(cv_labels, dtype=torch.float).unsqueeze(1).detach()


# Train models & print metrics on training data

# Neural net
# nn_model = train_model(train_features_tensor, train_labels_tensor)
# training_nn_predictions = make_torch_predictions(nn_model, train_features_tensor)
# print_metrics(predictions=training_nn_predictions, labels=train_labels_tensor,
#               model_name='Neural Network (training data)')
# cv_nn_predictions = make_torch_predictions(nn_model, cv_features_tensor)
# print_metrics(predictions=cv_nn_predictions, labels=cv_labels_tensor,
#               model_name='Neural Network (cross-validation data)')

# Random forest
# Try with default parameters
# forest_model = RandomForestClassifier(random_state=1)
# forest_model.fit(train_features, train_labels)
# training_forest_predictions = forest_model.predict(train_features)
# cv_forest_predictions = forest_model.predict(cv_features)
# print_metrics(predictions=training_forest_predictions, labels=train_labels, model_name='Random Forest (training data)')
# print_metrics(predictions=cv_forest_predictions, labels=cv_labels, model_name='Random Forest (cross-validation data)')

# search_forest_params(train_features_array, train_labels_array)
# Model we found in hyperparameter search
best_forest_model = RandomForestClassifier(criterion='entropy', max_depth=7, n_estimators=64, random_state=1)
best_forest_model.fit(train_features_array, train_labels_array)
print('RandomForest cross-validation score on training data:',
      cross_val_score(best_forest_model, train_features_array, train_labels_array).mean())

# Generate submissions
# generate_nn_submission(train_features_array, train_labels_array, features_for_training)
generate_forest_submission(best_forest_model, relevant_features=features_for_training, to_normalize=to_normalize)



