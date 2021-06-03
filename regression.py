import pandas as pd
import json

################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################


def parse_col_vals(entry):
    ret_val = ''
    for obj in json.loads(entry.replace('\'', '\"')):
        for k, v in obj.items():
            if k != 'id':
                ret_val += v + '*'
    return ret_val[:-1] if ret_val != '' else ret_val


def pre_process(path):
    """
    this function pre-processes the data samples in the given path (path should be a csv file)
    """
    df = pd.read_csv(path)
    json_cols = ['genres']

    # col_items = [item for item in df.pop('genres')]
    for col in json_cols:
        for i in range(df.shape[0]):
            df.loc[i, col] = parse_col_vals(df.loc[i, col])
        df = pd.concat([df, df[col].str.get_dummies(sep='*')], axis=1)
        df.drop(col, axis=1)
    return df


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


if __name__ == '__main__':
    pre_process('./movies_dataset.csv')