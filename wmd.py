from random import choice, randint

import numpy as np

from gensim.corpora.dictionary import Dictionary
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords

class WMDComputer:

    def __init__(self, w2v_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print('Loading Word2Vec...')
        self.w2v = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        print('Done')


    def wmd(self, doc1, doc2):
        '''
        Gensim wmdistance method. To compare compare to other ways of computing
        wmd.

        doc1, doc2: strings to compare

        returns: word moving distance
        '''

        doc1 = self._preprocess(doc1)
        doc2 = self._preprocess(doc2)

        return self.w2v.wmdistance(doc1, doc2)


    def wcd(self, doc1, doc2):
        '''
        Word centroid distance.

        doc1, doc2: strings to compare

        returns: word centroid distance
        '''

        # Preprocess documents
        doc1 = self._preprocess(doc1)
        doc2 = self._preprocess(doc2)

        # Compute nBow representation
        d1, dict1 = self._compute_nBow(doc1)
        d2, dict2 = self._compute_nBow(doc2)

        # Get word vectors
        doc1_w2v = [self.w2v[token] for token in dict1.values()]
        doc2_w2v = [self.w2v[token] for token in dict2.values()]

        # Compute weigihted average word vectors
        Xd1 = np.mean(d1 @ doc1_w2v)
        Xd2 = np.mean(d2 @ doc2_w2v)

        # Return euclidian distance between weighted average word vectors
        return np.linalg.norm(Xd1 - Xd2)


    def rwmd(self, doc1, doc2):
        '''
        Relaxed word moving distance.

        doc1, doc2: strings to compare

        returns: relaxed word moving distance
        '''

        # Preprocess documents
        doc1 = self._preprocess(doc1)
        doc2 = self._preprocess(doc2)

        # Compute nBow representation
        d1, dict1 = self._compute_nBow(doc1)
        d2, dict2 = self._compute_nBow(doc2)

        # Get word vectors
        doc1_w2v = [self.w2v[token] for token in dict1.values()]
        doc2_w2v = [self.w2v[token] for token in dict2.values()]

        # Compute nearest neighbors 
        distances_1 = [np.inf] * len(doc1_w2v)
        distances_2 = [np.inf] * len(doc2_w2v)
        for i, vec1 in enumerate(doc1_w2v):
            for j, vec2 in enumerate(doc2_w2v):
                dist = np.linalg.norm(vec1 - vec2)
                if dist < distances_1[i]:
                    distances_1[i] = dist
                if dist < distances_2[j]:
                    distances_2[j] = dist

        # Computing relaxed solutions
        l1 = np.sum(d1 @ distances_1)
        l2 = np.sum(d2 @ distances_2)

        return max(l1, l2)


    def base_knn(self, query_document, documents, k):
        '''
        Compute k-nearest-neighbors using only wmd. To compare to other knn
        methods.

        query_document: string
        documents: list of strings to compare to query_document
        k: k-nearest-neihbors to keep

        returns: k-nearest-neigbors of query_document
        '''

        wmd = [self.wmd(query_document, doc) for doc in documents]
        return np.array(documents)[np.argsort(wmd)[:k]]


    def prefetch_prune_knn(self, query_document, documents, k):
        '''
        Prefetch and prune documents using wcd and rwmv to compute
        k-nearest-neighbors of query document.

        query_document: string
        documents: list of strings to compare to query_document
        k: k-nearest-neihbors to keep

        returns: k-nearest-neigbors of query_document
        '''

        # Prefetch k documents with smallest wcd
        wcd_documents = [self.wcd(query_document, doc) for doc in documents]
        wcd_sorted = np.argsort(wcd_documents)

        # Compute wmd on prefetched documnents
        argknn = list(wcd_sorted[:k])
        knn = [self.wmd(query_document, documents[i]) for i in argknn]

        max_nn = max(knn)
        # Prune documents based on rwcd
        for i in wcd_sorted[k:]:
            if self.rwmd(query_document, documents[i]) < max_nn:
                wmd = self.wmd(query_document, documents[i])
                if wmd < max_nn:
                    idx = np.argmax(knn)
                    knn[idx] = wmd
                    argknn[idx] = i
                    max_nn = max(knn)

        sort = np.array(sorted(zip(knn, argknn), key=lambda pair: pair[0]))
        return np.array(documents)[sort[:, 1].astype(np.int)]


    def _preprocess(self, doc):
        '''
        Preprocess document. Lower case, remove stopwords, remove out of
        vocabulary words.

        doc: string

        returns: preprocessed string
        '''

        #doc = remove_stopwords(doc.lower())
        return [token for token in doc.split() if token in self.w2v]


    def _compute_nBow(self, doc):
        '''
        Compute normalized bag of word representation of a document.

        doc: string

        returns: nBow representation
        '''

        dictionary = Dictionary(documents=[doc])
        nBow = np.zeros((len(dictionary), len(dictionary)))
        bow = dictionary.doc2bow(doc)
        for idx, freq in bow:
            nBow[idx, idx] = freq / len(doc)

        return nBow, dictionary

    def generate_random_docs(self, n):
        '''
        Generates n random documents from w2v vocabulmary

        n: number of documents to generate
        '''

        vocab = list(self.w2v.vocab.keys())

        return [
            ' '.join([choice(vocab) for _ in range(randint(100, 1000))])
            for _ in range(n)    
        ]

