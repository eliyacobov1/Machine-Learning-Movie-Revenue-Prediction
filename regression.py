import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import Perceptron, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import json
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

################################################
#
#     Task 1 - predict movies revenue & ranking
#
################################################

SEPARATOR = '*'
NUM_TOP_OBSERVED_ELEMENTS = 20


def parse_col_vals(entry):
    """
    this function receives a data-frame entry of json form and converts it into an asterisk separated string
    for instance: [{'id': 12, 'name': 'Action'}, {'id': 13, 'name': 'Horror'}] -> 'Action*Horror'
    """
    if type(entry) != str and math.isnan(entry):
        return
    ret_val = ''
    try:
        entry_objects = json.loads(entry.replace('\"', '').replace('\'', '\"'))
    except json.decoder.JSONDecodeError:
        entry_objects = eval(entry)
    entry_objects = entry_objects if type(entry_objects) == list else [entry_objects]
    for obj in entry_objects:
        for k, v in obj.items():
            if k == 'name':
                ret_val += v + SEPARATOR
                break
    return ret_val[:-1] if ret_val != '' else ret_val


def col_top_appearance_count(col: pd.Series):
    col = col.dropna()
    col_as_list = SEPARATOR.join(col.tolist()).split(SEPARATOR)
    col_element_count = {val: col_as_list.count(val) for val in set(col_as_list)}
    return dict(Counter(col_element_count).most_common(NUM_TOP_OBSERVED_ELEMENTS))


def budget_average(data):
    """
    this function get budget column and replace all the index with the 0 budget to the average of all the other
    index with budget, we doing that cause we believe that if we have a lot of movies without budget it will
    cause to a lot of noise.
    :param data: in panda
    :return: fix budget column in pandas
    """
    budget = data['budget']
    temp = (budget != 0)
    average = budget[temp].to_numpy().mean()
    return budget.replace(0, average)


class PredictionModel:
    def __init__(self):
        self.model = GradientBoostingRegressor()
        self.scalar = StandardScaler()
        self.train_data, self.test_set, self.text_model, self.text_vectorizer, self.response_vec, \
        self.categorical_json_cols = None, None, None, None, None, None

    def pre_process(self, data, mode='train'):
        data.dropna(subset=['release_date'], inplace=True)
        release_date = data['release_date']

        release_month = release_date.str.split('/', expand=True)[1]
        release_year = release_date.str.split('/', expand=True)[2]

        release_year = release_year.apply(lambda x: 'NONE' if (type(x) != str) else
        ('80s' if (1980 > int(x) >= 1970) else ('90s' if (1990 > int(x) >= 1980) else
                                                ('2000s' if (2000 > int(x) >= 1990) else (
                                                    '2010s' if (2010 > int(x) >= 2000) else
                                                    ('2020s' if (2020 > int(x) >= 2010) else 'NONE'))))))

        release_month_dummies = pd.get_dummies(release_month)
        release_year_dummies = pd.get_dummies(release_year)
        release_year_dummies.drop(['NONE'], axis='columns', inplace=True)
        data = data.join(release_year_dummies).join(release_month_dummies)

        data.drop(['id', 'original_title', 'homepage'], axis='columns', inplace=True)
        data.dropna(
            subset=['budget', 'overview', 'vote_average', 'vote_count', 'production_companies', 'production_countries',
                    'release_date', 'runtime', 'spoken_languages',
                    'status', 'title', 'cast', 'crew', 'revenue', 'original_language', 'genres'], inplace=True)

        data['original_language'] = data['original_language'].apply(lambda x: 1 if (x == "en") else 0)
        data['status'] = data['status'].apply(lambda x: 1 if (x == "Released") else 0)
        data['budget'] = budget_average(data)

        data.drop(['overview', 'release_date', 'spoken_languages', 'title', 'cast', 'crew', 'tagline'], axis='columns',
                  inplace=True)

        # -------------------------- handle genres and collection categorical variable --------------------------
        categorical_json_cols = ['genres']
        for col in categorical_json_cols:
            data[col] = data[col].apply(parse_col_vals)
            data = pd.concat([data, data[col].str.get_dummies(sep=SEPARATOR)], axis=1)
            data = data.drop(columns=[col])

        # ---------------- handle categorical variables by creating a column for Top 10 values ----------------
        if mode == 'train':
            self.categorical_json_cols = {col: {} for col in
                                          ('belongs_to_collection', 'production_companies', 'production_countries',
                                           'keywords')}
        for col in self.categorical_json_cols.keys():
            data[col] = data[col].apply(parse_col_vals)
            if mode == 'train':
                self.categorical_json_cols[col] = col_top_appearance_count(data[col])
            new_col = data[col].apply(
                lambda s: len(set(s.split(SEPARATOR)).intersection(set(self.categorical_json_cols[col])))
                if type(s) == str else 0
            )
            new_col = new_col.rename(f"Top{NUM_TOP_OBSERVED_ELEMENTS} {col}")
            data = pd.concat([data, new_col], axis=1)
            data = data.drop(columns=[col])

        return data

    def pre_process_training_data(self):
        self.train_data = self.pre_process(self.train_data)

    def test_data_pre_process(self):
        self.test_set = self.pre_process(self.test_set, mode='test')

    def print_feature_correlation(self, data: pd.DataFrame):
        for feature in data.columns:
            if feature != 'revenue':
                print(f"feature {feature} is with correlation {self.train_data[feature].corr(self.response_vec)}")

    def fit(self, path):
        self.train_data = pd.read_csv(path)
        self.pre_process_training_data()
        self.response_vec = self.train_data['revenue']
        self.train_data = self.train_data.drop(columns=['revenue'])
        # self.text_model, self.text_vectorizer = self.train_text_model()
        self.model.fit(self.scalar.fit_transform(self.train_data), self.scalar.fit_transform(self.response_vec.to_numpy().reshape(-1, 1)))

    def predict(self, csv_file):
        """
        This function predicts revenues and votes of movies given a csv file with movie details.
        Note: Here you should also load your model since we are not going to run the training process.
        :param csv_file: csv with movies details. Same format as the training dataset csv.
        :return: a tuple - (a python list with the movies revenues, a python list with the movies avg_votes)
        """
        self.test_set = pd.read_csv(csv_file)
        self.test_data_pre_process()
        df = pd.DataFrame(0, index=np.arange(self.test_set.shape[0]), columns=self.train_data.columns)
        df.update(self.test_set)
        print(mean_squared_error(self.model.predict(self.scalar.fit_transform(df)), self.scalar.fit_transform(self.test_set['revenue'].to_numpy().reshape(-1, 1))))

    def train_text_model(self):
        """
        this function trains a model for text-overview score generation
        """
        data = self.train_data
        x_train = data['overview']
        y_train = data['revenue']
        x_train = x_train.to_numpy()
        y_train = y_train.to_numpy()

        free_text_vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', stop_words={'english'})
        x_train = free_text_vectorizer.fit_transform(x_train.tolist())
        model = LinearRegression()
        model.fit(x_train, y_train)
        return model, free_text_vectorizer


if __name__ == '__main__':
    m = PredictionModel()
    m.fit('./movies_dataset.csv')
    m.predict('./movies_dataset_part2.csv')
    # m.train_data.to_csv('pre_processed_dataset.csv')
    z = 0
