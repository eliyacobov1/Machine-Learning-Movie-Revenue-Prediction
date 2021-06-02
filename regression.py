import pandas as pd


################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

def pre_process(path):
    """
    this function pre-processes the data samples in the given path (path should be a csv file)
    """
    data = pd.read_csv(path)
    data.join(data.pop('genre').str.join('|').str.get_dummies())
    # filters = (1 <= data['condition']) & (data['condition'] <= 5)  # condition within valid range
    # filters = filters & data['date'].notnull() & data['id'].notnull()  # essential fields date and id are present
    # filters = filters & (data['price'] > 0) & (data['sqft_living'] > 0) \
    #           & (data['floors'] > 0) & (data['sqft_lot15'] > 0)  # non-negative values on corresponding fields
    data = data.loc[filters]  # apply conditions
    data = data.drop_duplicates()  # drop duplicates
    # separate year and month and convert them into dummy-variables
    sliced_dates = pd.concat([data['date'].str.slice(4, 6), data['date'].str.slice(0, 4)], axis=1)
    data = pd.concat([data, pd.get_dummies(sliced_dates.iloc[:, 0]), pd.get_dummies(sliced_dates.iloc[:, 1])], axis=1)
    data = data.drop(['id'], axis=1)  # drop categorical features
    return data


def fit(path):
    pass


def predict(csv_file):
    """
    This function predicts revenues and votes of movies given a csv file with movie details.
    Note: Here you should also load your model since we are not going to run the training process.
    :param csv_file: csv with movies details. Same format as the training dataset csv.
    :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
    """

    pass
