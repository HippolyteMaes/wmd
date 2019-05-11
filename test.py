import time

import wmd


def test_distances(computer, pairs):
    for doc1, doc2 in pairs:
        start = time.time()
        wmd = computer.wmd(doc1, doc2)
        wmd_t = time.time()
        wcd = computer.wcd(doc1, doc2)
        wcd_t = time.time()
        rwmd = computer.rwmd(doc1, doc2)
        rwmd_t = time.time()

        print('{} | {}'.format(doc1, doc2))
        print('WCD -> ', wcd, ': %f' % (wcd_t - wmd_t))
        print('RWCD -> ', rwmd, ': %f' % (rwmd_t - wcd_t))
        print('WMD -> ', wmd, ': %f' % (wcd_t - start))
        assert wcd <= wmd
        assert rwmd <= wmd


def test_knns(computer, query_documents, documents):
    for query_document in query_documents:
        k = 3

        res = []
        for method in [computer.base_knn, computer.prefetch_prune_knn]:
            start = time.time()
            knns = method(query_document, documents, k)
            end = time.time()
            print('{} | {} - {}'.format(method.__name__,
                                        query_document,
                                        end - start))
            assert len(knns) == k
            assert all([
                (computer.wmd(query_document, knns[i])
                 <= computer.wmd(query_document, knns[i + 1]))
                for i in range(len(knns) - 1)
            ])
            res.append(knns)

        assert (res[0] == res[1]).all()



if __name__ == '__main__':

    computer = wmd.WMDComputer('GoogleNews-vectors-negative300.bin')

    pairs = [
        ('dog', 'cat'),
        ('Zero distance test', 'Zero distance test'),
        ('The president eat ice cream', 'The director enjoys snacks'),
        ('I walk away', 'Beautiful building'),
        ('I walk far far away', 'Beautiful building'),
        ('Obama speaks to the media in Illinois',
         'The president greets the press in Chicago'),
        ('Obama speaks in Illinois',
         'The president greets the press in Chicago'),
        ('The band gave a concert in Japan',
         'The president greets the press in Chicago'),
        ('Zero distance test', 'The president greets the press in Chicago')
    ]

    documents = [
        'Zero distance test',
        'The president eat ice cream',
        'The director enjoys snacks',
        'I walk away',
        'Beautiful building',
        'I walk far far away',
        'Obama speaks to the media in Illinois',
        'The president greets the press in Chicago',
        'Obama speaks in Illinois',
        'The band gave a concert in Japan',
        'The goat eat grass in a field but the cow is jealous and tries to steal it from its friend'
    ]

    query_documents = [
        'Zero distance test',
        'The president is Obama',
        'A boat may land in Japan soon',
        'I have not eaten in ten days'
    ]

    # Testing distance methods
    test_distances(computer, pairs)

    # Testing knn methods
    test_knns(computer, query_documents, documents)

    # Testing knns methods on  a bigger scale
    random_docs = computer.generate_random_docs(100)
    test_knns(computer, query_documents, random_docs)
