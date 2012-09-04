How to run the program

COMMAND LINE

$ hw1.py [weighted?] [k] [metric] [mode] [querystring]


ARGUMENTS

weighted - specify weight options. It takes the following values
           0  for unweighted version
           1  for weighted version
           >1 for weighted version with normalized training set
              (for problem 3h)

k - specify number of neighbor to consider

metric - similarity metric
         0 for inverse Euclidean distance
         1 for dot product
         2 for cosine distance
         
mode - specify query mode (optional). It takes the following values
       u for user query mode 
         (will show top 10 recommendation for given user-id)
       a for artist query mode 
         (will show top 10 recommendation for people who like given artist)
       
querystring - specify user-id or artist name according to the mode


EXAMPLES

$ 0 1 0              # evaluate recommendation of all users using 
                     # unweighted version with k=1 and inverse euclidean distance
                     
$ 1 10 1 u 1000      # query user-id=1000 using weighted version with k=10 and dot product

$ 1 100 2 a maroon   # query artist="maroon" using weighted version with k=100 and cosine product

$ 2 10 2             # with normalized training set, 
                     # evaluate recommendation of all users using
                     # weighted version with  k=10 and cosine distance. 