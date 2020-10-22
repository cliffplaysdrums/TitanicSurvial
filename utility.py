import pandas as pd
from bokeh.plotting import figure, output_file, show
import numpy as np

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
    # 4: convert_sex,
    # 10: convert_cabin,
    11: convert_embarked
}


def create_two_feature_plot(df, feat1, feat2, feat1_range=None):
    output_file(f'plots/{feat1}-{feat2}.html')

    survivor_indices = df['Survived'] == 1
    x_survivors = df.loc[survivor_indices, feat1].values
    x_casualties = df.loc[~survivor_indices, feat1].values
    # Add some jitter so we don't have too many points overlapping and can see them all
    if feat1_range is not None:
        x_survivors = list(zip(x_survivors, np.random.normal(scale=.05, size=len(x_survivors))))
        x_casualties = list(zip(x_casualties, np.random.normal(scale=.05, size=len(x_casualties))))

    p = figure(title=f'Titanic Passenger Survival', x_range=feat1_range, x_axis_label=feat1, y_axis_label=feat2)
    p.scatter(x_survivors, df.loc[survivor_indices][feat2], size=4, color='#3A5785', legend_label='Survivors')
    p.scatter(x_casualties, df.loc[~survivor_indices][feat2], size=4, color='#B5A87A', legend_label='Casualties')
    show(p)


def get_training_data():
    return pd.read_csv('data/train.csv', converters=TITANIC_DATA_CONVERTERS)


df = get_training_data()
create_two_feature_plot(df, 'Sex', 'Age', feat1_range=['male', 'female'])
