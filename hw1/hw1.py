#!/usr/bin/python
import math
import sys
import time
import random

# important variables
# x_u      - training set
# y_u      - test set
# weighted - 0 if unweighted, otherwise weighted
# k        - no. of neighbor to consider
# Kfn      - function pointer to the specifed similarity metric
def queryAll(x_u, y_u, song_map, weighted, k, Kfn):
    precision = 0.0;   
    precision_random = 0.0;
    precision_top10hits = 0.0;
    
    # count number of plays for each songs (to create popularity-based baesline)
    tophits = {}
    for user in x_u:
        for song in x_u[user]:
            tophits[song] = tophits[song]+x_u[user][song] if song in tophits else x_u[user][song]
    tophits = sorted(tophits,key=tophits.get, reverse=True)
    
    start = time.clock()
    for user in x_u:
        # calculate k nearest neighbor
        n_k,similarity = calculateNearestNeighbor(x_u,x_u[user],k,Kfn)
        # calculate ranking vector
        r_rel = calculateRankingVector(x_u, x_u[user], weighted, n_k, similarity)
        hit = [x for x in r_rel if x in y_u[user]]        
        precision += len(hit)/10.0;
        
        # random baseline
        songlist = list(set(song_map.keys()) - set(x_u[user]))
        random.shuffle(songlist)
        hit = [x for x in songlist[:10] if x in y_u[user]]
        precision_random += len(hit)/10.0
        
        # popularity baseline
        top10hits = []
        for song in tophits:
            if song not in x_u[user]:
                top10hits.append(song)
                if len(top10hits) == 10:
                    break    
        hit = [x for x in top10hits if x in y_u[user]]
        precision_top10hits += len(hit)/10.0
    
    print("Got result with precision=%0.2f in %0.2f sec" % (precision/(len(x_u)-1), time.clock() - start) )
    print("Random baseline precision=%0.2f" % (precision_random/(len(x_u)-1)))
    print("Popularity baseline precision=%0.2f" % (precision_top10hits/(len(x_u)-1)))

def queryUser(x_u, y_u, song_map, weighted, k, Kfn, userParam):
    start = time.clock()
    # calculate k nearest neighbor
    n_k,similarity = calculateNearestNeighbor(x_u,x_u[userParam],k,Kfn)
    # calculate ranking vector
    r_rel = calculateRankingVector(x_u, x_u[userParam], weighted, n_k, similarity)
    
    print("User Playlist's Top 10")
    printFirstTenSongs(sorted(x_u[userParam], reverse=True), song_map)

    hit = [x for x in r_rel if x in y_u[userParam]]    
    print("Top 10 Recommendations (precision=%0.1f in %0.2f sec)" % (len(hit)/10.0,time.clock()-start))
    printFirstTenSongs(r_rel, song_map)
    
def queryArtist(x_u, y_u, song_map, weighted, k,  Kfn, artistParam):
    start = time.clock()
    # create a list of all song from this artist
    x_u_artist = {}
    for user in x_u:
        for song in x_u[user]:
            if song_map[song].lower().find(artistParam.lower()) != -1:
                x_u_artist[song] = 1
    #print("done artist vector:    %0.2f sec since started" % (time.clock()-start) )
    # calculate k nearest neighbor
    n_k,similarity = calculateNearestNeighbor(x_u,x_u_artist,k,Kfn)
    #print("done nearest neighbor: %0.2f sec since started" % (time.clock()-start) )
    # calculate ranking vector
    r_rel = calculateRankingVector(x_u, x_u_artist, weighted, n_k, similarity)
    print("Top 10 Recommendations for those who liked '%s' (in %0.2f sec)" % (artistParam, time.clock()-start))
    printFirstTenSongs(r_rel, song_map)
    
def calculateNearestNeighbor(x_u, userFeature ,k , Kfn):
    similarity = {} # between parameter and everyone
    # calculate similarity metric
    for user in [x for x in x_u if x_u[x] != userFeature ]:
        similarity[user] = Kfn(userFeature,x_u[user])
    # sort to get nearest neighbor
    Nk = sorted(similarity,key=similarity.get, reverse=True)[:k]
    return Nk,similarity

def calculateRankingVector(x_u, userFeature, weighted, n_k, similarity):
    r_u = {}
    # calculate Ru
    for kth_near, user in enumerate(n_k, start=1): # each kth nearest neighbor
        for song in [ x for x in x_u[user] if x not in userFeature ]: # each song excluding exising
            if song not in r_u:
                r_u[song] = 0
            if weighted == 0:
                r_u[song] += x_u[user][song]/kth_near # unweighted
            else:
                r_u[song] += x_u[user][song]*similarity[user]  # weighted    
    # sort and return top ten then evaluate        
    r_rel = sorted(r_u,key=r_u.get, reverse=True)[:10]
    return r_rel

def normalize(x_u,song_map):
    # implement the idea from chapter 9 of Mining of Massive Datasets 
    # http://i.stanford.edu/~ullman/mmds.html
    print("Normalize = ON for problem 3(h)")
    for user in x_u:
        # minus average
        avg = sum(x_u[user].values())/len(x_u[user].values())
        for song in x_u[user]:
            x_u[user][song] -= avg
    return x_u
# --------------------
# 3 similarity metrics
# --------------------
def inverseEuclid(a,b):
    aKey = set(a.keys())
    bKey = set(b.keys())
    sumDiffSq = sum( [ (a[x]-b[x])**2 for x in list(aKey&bKey) ] )
    sumDiffSq+= sum( [ a[x]**2 for x in list( aKey - bKey ) ] )
    sumDiffSq+= sum( [ b[x]**2 for x in list( bKey - aKey ) ] )    
    return 1/math.sqrt(sumDiffSq) if sumDiffSq != 0 else 2**32

def dotProduct(a,b):  
    aKey = set(a.keys())
    bKey = set(b.keys())
    return sum( [a[x]*b[x] for x in list(aKey&bKey)] )
 
def cosDistance(a,b):  
    l2a = sum( [a[x]**2 for x in a.keys()] )
    l2b = sum( [b[x]**2 for x in b.keys()] )
    return dotProduct(a,b) / (l2a*l2b) if l2a*l2b != 0 else 0

# -----------------------------------
# other misc. utilities (not printed)
# -----------------------------------
def readUserTrain():
    x_u = {}
    file = open("user_train.txt","r")
    for line in file:
        features = line.split(' ')
        x_u[int(features[0])] = {}
        for feature in features[2:]:
            song,rating = feature.split(':')
            x_u[int(features[0])][int(song)] = int(rating)
    file.close()
    return x_u

def readUserTest():
    y_u = {}
    file = open("user_test.txt","r")
    for line in file:
        favorites = line.split(' ')
        y_u[int(favorites[0])] = [int(x) for x in favorites[2:]]
    file.close()
    return y_u    

def readSongMap():
    song_map = {}
    file = open("song_mapping.txt","r")
    for line in file:
        token = line.split('\t')
        song_map[int(token[0])] = token[1].strip() + " - " + token[2].strip()
    file.close()
    return song_map       

def printFirstTenSongs(l,song_map):
    for i,song in enumerate(l[:10], start=1):
        print("  %2d. %s" % ( i, song_map[song] ))
    print("")

# program entry point
def main():    
    # initializations
    K = {0: inverseEuclid, 1: dotProduct, 2: cosDistance}
    x_u = readUserTrain()
    y_u = readUserTest()
    song_map = readSongMap()
    
    userInput = sys.argv[1:]
    
    if len(userInput) <3:
        print("Too few arguments.")
        sys.exit()
    
    weighted = int(userInput[0])
    k = int(userInput[1])
    metric = int(userInput[2])
    
    if weighted > 1:
        x_u = normalize(x_u, song_map)
    
    if len(userInput) == 3:
        queryAll(x_u, y_u, song_map, weighted, k, K[metric])
    elif len(userInput) == 5 and userInput[3] == 'u':        
        queryUser(x_u, y_u, song_map, weighted, k, K[metric], int(userInput[4]))
    elif len(userInput) == 5 and userInput[3] == 'a':        
        queryArtist(x_u, y_u, song_map, weighted, k, K[metric], userInput[4])
    else:
        print("Wrong argument format. Here are examples:")
        print('  "1 1 1"         (weighted, k=1, metric=dot product)')
        print('  "0 3 2 u 1000"  (unweighted, k=3, metric=cosine, query userid=1000)')
        print('  "1 5 0 a bruno" (weighted, k=5, metric=inverse euclidean, query artist=bruno)')
        print('  "2 10 2"        (weighted, k=10, metric=cosine, with training set normalization (weight=2)')

# standard boilerplate from google Python tutorial    
if __name__ == '__main__':
    main()

# test cases
"""
testA = { "i":3, "j":4, "k":5}
testB = { "i":3, "j":4, "k":5}
print("inv " + str(inverseEuclid(testA,testB)) )
print("dot " + str(dotProduct(testA,testB)) )
print("cos " + str(cosDistance(testA,testB)) )
"""
"""
testA = x_u[1]
testB = x_u[3323]
print("inv " + str(inverseEuclid(testA,testB)) )
print("dot " + str(dotProduct(testA,testB)) )
print("cos " + str(cosDistance(testA,testB)) )
"""    
"""
print([str(x) for x in Nk])
print([str(similarity[x]) for x in Nk])
"""
"""
print(r_rel)
print([r_u[x] for x in r_rel])
"""