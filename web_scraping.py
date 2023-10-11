from requests_html import HTMLSession
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
#nltk.download('punkt')

def get_parser(soup, review_list, overall_rating_list):
    reviews = soup.find_all('span', {'data-testid':'wrapper-tag', 'class':'t9JcvSL3Bsj1lxMSi3pz h_kb2PFOoyZe1skyGiz9 DUkDy8G7CgNvYcWgJYPN'}) 
    ratings = soup.find_all('span', {'class':'Q2DbumELlxH4s85dk8Mj'})
    for review in reviews:  # reviews is a list of items, need to use loop to process all searched items
        review = review.get_text(separator=' ')  # get text(comment) from each item
        review_list.append(review)
    for rating in ratings[::4]: 
        rating = rating.get_text(separator=' ')
        overall_rating_list.append(rating)

def get_total_pages(url):
    session = HTMLSession()
    response = session.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    soup_find_total_page = soup.find_all('script') 
    words = word_tokenize(str(soup_find_total_page[-3]))
    l = []
    flag = 0
    for i in words:
        if i =='totalPages':
            flag = 1
        if flag == 1:
            l.append(i)
    total_pages = int(l[2].replace(":", ""))
    return total_pages

def get_reviews_from_all_pages(first_page_url, max_page):
    page = 1
    review_list = []
    overall_rating_list = []
    p1, p2 = first_page_url.split('page=')

    while page!=(max_page+1):
        url = p1 + 'page=' + str(page)+ p2[1:]
        session = HTMLSession()
        response = session.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        get_parser(soup, review_list, overall_rating_list)        
        page += 1
    df = pd.DataFrame({'review': np.asarray(review_list), 'overall rating': np.asarray(overall_rating_list)})
    df['overall rating'] = df['overall rating'].astype(int) - 1  # label to ID, {'5 stars':4, '4 stars':3, ... , '1 stars': 0} 
    return df

def get_data_from_page_list(url_list):
    df_list = []
    for url in url_list:
        max_page = get_total_pages(url)
        df = get_reviews_from_all_pages(url, max_page)
        df_list.append(df)
    df = pd.concat(df_list)
    return df
        
if __name__ == '__main__':

    url_list = [
                # rating above 4
                'https://www.opentable.ca/miller-tavern?originId=9ebce773-3b2e-48fc-93bc-8b79a06e95bc&corrid=9ebce773-3b2e-48fc-93bc-8b79a06e95bc&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&page=1&sortBy=newestReview',
                'https://www.opentable.ca/r/blue-blood-steakhouse-toronto?originId=1937513f-6bbc-4f51-b20e-444a14fea337&corrid=1937513f-6bbc-4f51-b20e-444a14fea337&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&page=1&sortBy=newestReview',
                # same restaurant but different location
                'https://www.opentable.ca/the-keg-steakhouse-and-bar-york-street?corrid=b977b24e-4643-4356-9441-763d4bebd7cf&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjoxLCJuIjowfQ&p=2&sd=2023-10-10T19%3A00%3A00&page=1&sortBy=newestReview',
                # rating 3.1
                'https://www.opentable.ca/r/chez-mal-manchester?page=1&sortBy=newestReview',
                # rating 2.9
                'https://www.opentable.ca/r/lookout-rooftop-boston?page=1&sortBy=newestReview',
                'https://www.opentable.ca/r/bar-31-shangri-la-the-shard-london?page=1&sortBy=newestReview',
                # rating 2.3
                'https://www.opentable.ca/pizza-rustica-restaurant-and-bar?originId=d084e009-f0b5-4a6f-8ba0-477c01aea935&corrid=d084e009-f0b5-4a6f-8ba0-477c01aea935&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjoxLCJuIjowfQ&p=2&sd=2023-10-10T19%3A00%3A00&page=1&sortBy=newestReview',
                # rating 1.9
                'https://www.opentable.ca/r/lime-an-american-cantina-denver?page=1&sortBy=newestReview',
                # rating 1.6
                'https://www.opentable.ca/r/bourgee-lakeside-grays?page=1&sortBy=newestReview',
                # rating 1.3
                'https://www.opentable.ca/r/chophouse-363-chino?page=1&sortBy=newestReview'
                ]

    eval_url = [
                'https://www.opentable.ca/r/the-keg-steakhouse-and-bar-north-york?originId=bcc0b7a5-d42e-468c-8a2d-985968665f45&corrid=bcc0b7a5-d42e-468c-8a2d-985968665f45&avt=eyJ2IjoyLCJtIjoxLCJwIjowLCJzIjowLCJuIjowfQ&page=1&sortBy=newestReview'
                ]

    df = get_data_from_page_list(url_list)
    eval = get_data_from_page_list(eval_url)
    df.to_csv('./train.csv', index=False)
    eval.to_csv('./eval.csv', index=False)