import multiprocessing as mp

import wikipedia

import config
from context import Context
from question import Question


class DocumentRetriever:
    def retrieve_questions(self, question):
        doc_names = self._search_docs(question.replace('"', ''))
        if len(doc_names) == 0:
            return []

        with mp.Pool(len(doc_names)) as p:
            docs = p.map(self._get_doc, doc_names)
        passages = sum(docs, [])

        with mp.Pool(config.WIKI_THREADS) as p:
            contexts = p.map(self._make_context, passages)
            contexts = [c for c in contexts if len(c.tokens) > 2]
        return Question.distribute(question, contexts)

    def _search_docs(self, question):
        doc_names = wikipedia.search(question)
        return doc_names[:config.WIKI_ARTICLE_COUNT]

    def _get_doc(self, doc_name):
        try:
            content = wikipedia.page(doc_name).content
        except Exception as e:
            print('Error retrieving document "{}": {}'.format(doc_name, e))
            return []
        return [(c, doc_name) for c in self._parse_content(content)]

    def _parse_content(self, content):
        lines = [line.strip() for line in content.split('\n')]
        passages = []
        for line in lines:
            if line == '== See also ==':
                break
            if line and not line.startswith('=='):
                passages.append(line)
        return passages

    def _make_context(self, passage):
        return Context(passage[0], doc_name=passage[1])
