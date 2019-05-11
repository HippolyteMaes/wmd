# wmd case study

Config
------
- Python 3.7
- numpy
- gensim
- GoogleNews Word2Vec

Pseudo code
-----------

```python

def WCD(doc1, doc2):
  preprocess(doc1)
  preprocess(doc2)

  d1 = compute_NBow(doc1)
  d2 = compute_NBow(doc2)
  
  Xd1 = mean(d1 * w2v(doc1))
  Xd2 = mean(d2 * w2v(doc2))

  return norm(Xd1 - Xd2)
  
  
def RWMD(doc1, doc2):
  preprocess(doc1)
  preprocess(doc2)
  
  d1 = compute_NBow(doc1)
  d2 = compute_NBow(doc2)

  distances_1to2, distances_2to1 = nearest_neighbor(doc1, doc2)

  l1 = sum(d1 * distances_1to2)
  l2 = sum(d2 * distances_2to1)
  return max(l1, l2)
  
  
def prefetch_and_prune(query, collection, k):
  wcds = list(foreach doc in collection do wcd(query, doc))
  knn = list(foreach doc in take_minimum_k(wcds) do wmd(query, doc)) 
  
  for doc in wcds:
    if rwmd(query, doc) < max(knn):
      insert(knn, wmd(query, doc))
      knn = take_minimum_k(knn)
      
  return sort(knn)
```

Examples of usage on sample sentences
-------------------------------------

See test.py


The time complexity of your implementation
------------------------------------------

With p the number of word in a document and d the number of documents

WCD - O(p)

RWMD - O(p^2)


prefetch and prune:

worst case - O(d * p^3 * log(p)) 

best case  - O(d * p^2)

Estimation of the speed gain over the current implementation of the standard WMD in gensim
-------------------------------------------------------------------------------------------

The should be a clear speed gain for big collections of relatively long documents.
We can take advantage of RWMD lower time complexity to save a lot of time pruning documents.
Bigger collections amplify the gains.
