# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:43:23 2022

@author: WiN 10
"""
import time
def get_tweet_by_link(url):
    import tweepy
    
    id_status = url.split('status/')[-1]
    #create tweepy's api
    consumer_key='96Rm00PfMEwtkBjhBoWlGwzDG'
    consumer_secret='DUdU7P0CpbOAb0whx5pq28qEWlMsZOk3d9DkNZHyvj1bYOPldU'
    
    access_token='1214597752098717696-ID8wKAYJgZ3H35ebMAFkVKV1py6w19'
    access_token_secret='AYrRow7HiU2EUcdQx6co7ZT7dqMnweFOtvOdxHhyJZJYZ'
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    #query
    status = api.get_status(id_status, tweet_mode="extended")
    content = status.full_text
    author = status.author.screen_name
    create_date = status.created_at
    user = api.get_user(screen_name=author)
    verified = user.verified
    
    # get the screen names of the retweeters and follower
    retweet_list = []
    for retweet in api.get_retweets(id=id_status):
        retweet_list.append(retweet.user.screen_name)
    
    follower_list = []
    for follower in api.get_followers(screen_name=author):
        follower_list.append(follower.screen_name)
        
    return([content,author,create_date,verified,retweet_list,follower_list])

start = time.time()
print(get_tweet_by_link("https://twitter.com/Offchainon/status/1483631620519501827"))
print('\ntimer %s seconds'%(time.time()-start))