import random

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


def generate_m3com_pageview():
    examination = 0.005
    df = pd.read_csv('true_relevance.csv')
    df['click'] = df['relevance'].apply(lambda relevance: np.random.binomial(1, relevance * examination))
    pageview = df[df['click'] == 1]
    pageview.to_csv('pageview.csv', index=False)


def generate_mail_magazine_pageview():
    examinations = [0.1, 0.05, 0.03, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004]
    df = pd.read_csv('true_relevance.csv')

    target_user_num = 20000
    target_users = np.arange(1, target_user_num + 1)
    df = df[df['user'].isin(target_users)]

    df_mails = []
    for user in range(1, target_user_num + 1):
        mail = random.sample(list(df['item'].unique()), 10)
        df_mail = pd.DataFrame({'user': [user] * 10, 'item': mail, 'examination': examinations})
        df_mails.append(df_mail)

        if user % 1000 == 0:
            print(user)
    df_mail = pd.concat(df_mails)

    df = pd.merge(df, df_mail, on=['user', 'item'], how='left')
    df = df.fillna(0)

    df['click_prob'] = df['relevance'] * df['examination']
    df['click'] = df['click_prob'].apply(lambda p: np.random.binomial(1, p))
    pageview = df[df['click'] == 1]
    pageview.to_csv('mail_magazine_pageview.csv', index=False)

def get_ranking():
    pageview = pd.read_csv('pageview.csv')
    import pdb; pdb.set_trace()

def get_mailmagazine_ranking():
    pageview = pd.read_csv('mail_magazine_pageview.csv')
    pageview['weighted_click'] = pageview['click'] / pageview['examination']
    df = pageview.groupby('item').agg({'weighted_click': 'sum'}).sort_values(by='weighted_click', ascending=False)
    import pdb; pdb.set_trace()


# generate_true_relevance()
# generate_m3com_pageview()
# get_ranking()

# generate_mail_magazine_pageview()
get_mailmagazine_ranking()