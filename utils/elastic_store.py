from elasticsearch import Elasticsearch, helpers
import os


class ElasticsearchWrapper:

    def __init__(self, host='localhost', port=443, index_name='documents'):
        self.host = host
        self.port = port
        self.index_name = index_name
        self.es = Elasticsearch(['http://localhost:9200'], basic_auth=('elastic', 'test1234'))
        self._create_index_if_not_exists()


    def _create_index_if_not_exists(self):
        if not self.es.indices.exists(index=self.index_name):
            self.es.indices.create(index=self.index_name)
            print(f"Index '{self.index_name}' created.")
        else:
            print(f"Index '{self.index_name}' already exists.")


    def index_document(self, doc_id, document_body):
        response = self.es.index(index=self.index_name, id=doc_id, body=document_body)
        return response


    def bulk_index_documents_from_dir(self, directory):
        success, failed = helpers.bulk(self.es, self._generate_docs_from_dir(directory))
        print(f"Successfully indexed {success} documents.")
        if failed:
            print(f"Failed to index {failed} documents.")


    def _generate_docs_from_dir(self, directory):
        for s in directory:
            yield {
                '_index': self.index_name,
                '_source': {
                    # 'filename': filename,
                    'content': s
                }
            }


    def search(self, query, size=10):
        response = self.es.search(index=self.index_name, body={
            'query': {
                'match': {
                    'content': query
                }
            },
            'size': size
        })
        return response


    def delete_index(self):
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
            print(f"Index '{self.index_name}' deleted.")
        else:            
            print(f"Index '{self.index_name}' does not exist.")


