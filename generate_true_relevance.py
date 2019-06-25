import pandas as pd
import numpy as np


def generate_true_relevance():
    item_num = 50
    user_num = 100000

    items = ['news_' + str(i) for i in range(1, item_num + 1)]
    item_strength = np.arange(0, 5, 0.1)

    df_item = pd.DataFrame({'item': items, 'item_strength': item_strength})
    df_item = pd.concat([df_item] * user_num).reset_index()[['item', 'item_strength']]

    users = np.arange(1, user_num + 1)
    df_user = pd.DataFrame({'user': users})
    df_user = pd.concat([df_user] * item_num).sort_values(by='user').reset_index()['user']

    df = pd.concat([df_item, df_user], axis=1)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    df['relevance'] = df['item_strength'].apply(lambda strength: sigmoid(strength + np.random.randn() - 2.5))

    df.to_csv('true_relevance.csv', index=False)


def generate_pageview():
    examination = 0.005
    df = pd.read_csv('true_relevance.csv')

    df['click'] = df['relevance'].apply(lambda relevance: np.random.binomial(1, relevance * examination))
    pageview = df[df['click'] == 1]

    pageview.to_csv('pageview.csv', index=False)


def get_ranking():
    pageview = pd.read_csv('pageview.csv')
    import pdb; pdb.set_trace()

# generate_true_relevance()
# generate_pageview()
# get_ranking()