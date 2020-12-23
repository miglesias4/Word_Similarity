#Created by: Dr. Olac Fuentes
#Modified by: Matthew Iglesias
import bs4 as bs
import urllib.request
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm

def read_embeddings(n=1000):
    # Reads n embeddings from file
    # Returns a dictionary were embedding[w] is the embeding of string w
    embedding = {}
    count = 0
    with open('glove.6B.50d.txt', encoding="utf8") as f: 
        for line in f: 
            ls = line.split(" ")
            emb = [np.float32(x) for x in ls[1:]]
            embedding[ls[0]]=np.array(emb) 
            count+=1    
            if count>= n:
                break
    return embedding

#Receives word w and embedding dictionary emb, returns word in E whose embedding is close to w
#if w is not in E: return '---' 
def most_similar(w,emb):
    
    try:
        for value in emb: #value is the actual word
            dist = np.linalg.norm(min(emb[w] - emb[value])) #calculate Euclidean distance
            if ((dist > 0.1) and (dist < 1.0)):
                return value
    except:
        pass
 
    return '---'
    
if __name__ == "__main__":  
    vocabulary_size = 30000        
    embedding = read_embeddings(vocabulary_size)
    
    for word in ['white', 'coyote', 'spain', 'football','taco','university','convolutional']:
        ms =  most_similar(word,embedding)
        print('The most similar word to {} is {}'.format(word,ms))
        
        
'''
The most similar word to white is black
The most similar word to coyote is wolf
The most similar word to spain is portugal
The most similar word to football is soccer
The most similar word to taco is burger
The most similar word to university is college
The most similar word to convolutional is ---
'''