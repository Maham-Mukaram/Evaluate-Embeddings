import os 
import pathlib 
import pickle 
from typing import List, Dict 

import cohere

class CohereEmbedder:
    """
    Class for embedding queries and corpus using Cohere.
    """
    def __init__(self, engine, task_name=None, batch_size=32, sep=" ", save_emb=False, API_KEY='C', base_path='', **kwargs):
        """
        Initialize the CohereEmbedder object.

        Parameters:
        - engine (str): Cohere embedding model.
        - task_name (str): Name of the task or dataset (used for saving embeddings).
        - batch_size (int): Batch size for processing queries or documents.
        - sep (str): Separator to concatenate title and text for document encoding.
        - save_emb (bool): Whether to save embeddings to files.
        - kwargs: Additional keyword arguments.
        """
        self.engine = engine
        self.batch_size = batch_size
        self.save_emb = save_emb
        self.base_path = base_path
        self.task_name = task_name
        self.sep = sep
        self.API_KEY = API_KEY

        if save_emb:
            assert self.task_name is not None

        # Create directory for saving embeddings if it doesn't exist
        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def encode_queries(self, queries: List[str], **kwargs):
        """
        Encode a list of queries using the Cohere API.

        Parameters:
        - queries (List[str]): List of query strings.
        - batch_size (int): Batch size for processing queries.
        - kwargs: Additional keyword arguments.

        Returns:
        - List: List of query embeddings.
        """

        fin_embeddings = []

        # Generate a unique filename for the query embeddings
        filename = f"{self.task_name}_{queries[0][:10]}_{queries[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        # Check if embeddings already exist in the file
        if queries and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            # If not, retrieve embeddings from Cohere API in batches
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i : i + self.batch_size]
                response = cohere.Client(self.API_KEY).embed(batch, input_type="search_query", model=self.engine).embeddings
                # May want to sleep here to avoid getting too many requests error
                fin_embeddings.extend(response)

        # Save embeddings to file
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        # Ensure the number of queries matches the number of embeddings
        assert len(queries) == len(fin_embeddings)
        return fin_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs):
        """
        Encode a list of documents in the corpus using the Cohere API.

        Parameters:
        - corpus (List[Dict[str, str]]): List of documents with title and text fields.
        - batch_size (int): Batch size for processing documents.
        - kwargs: Additional keyword arguments.

        Returns:
        - List: List of document embeddings.
        """
        if type(corpus) is dict:
            # If corpus is a dictionary, combine title and text
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            # If corpus is a list of documents, combine title and text
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        fin_embeddings = []

        # Generate a unique filename for the query embeddings
        filename = f"{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        # Check if embeddings already exist in the file
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            # If not, retrieve embeddings from Cohere API in batches
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]
                response = cohere.Client(self.API_KEY).embed(batch, input_type="search_document", model=self.engine).embeddings
                # May want to sleep here to avoid getting too many requests error
                fin_embeddings.extend(response)

        # Save embeddings to file
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        # Ensure the number of sentences matches the number of embeddings
        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings