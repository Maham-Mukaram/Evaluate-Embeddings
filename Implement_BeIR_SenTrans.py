import argparse
import json
import logging
import os
import pathlib
import pickle
import time
from typing import List, Dict

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from sentence_transformers import SentenceTransformer

# Configure logging for the script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of tasks to be processed
TASK_LIST = [
    "arguana",
    "cqadupstack/android",
    "cqadupstack/english",
    "cqadupstack/gaming",
    "cqadupstack/gis",
    "cqadupstack/mathematica",
    "cqadupstack/physics",
    "cqadupstack/programmers",
    "cqadupstack/stats",
    "cqadupstack/stats",
    "cqadupstack/tex",
    "cqadupstack/unix",
    "cqadupstack/webmasters",
    "cqadupstack/wordpress",
    "fiqa",
    "nfcorpus",
    "scidocs",
    "scifact",
    "webis-touche2020",
    "trec-covid"
]


class Embedder:
    """
    Class for encoding queries and corpus using SentenceTransformer.
    """
    def __init__(self, engine, task_name=None, batch_size=64, sep=" ", save_emb=False, **kwargs):
        """
        Initialize the Embedder with the specified parameters.

        Args:
        - engine: Model engine for SentenceTransformer.
        - task_name: Optional task name.
        - batch_size: Batch size for encoding.
        - sep: Separator for combining title and text.
        - save_emb: Flag indicating whether to save embeddings.
        """
        self.engine = SentenceTransformer(engine)
        self.batch_size = batch_size
        self.save_emb = save_emb
        self.base_path = f"BeIR-Embeddings/{engine.split('/')[-1]}/"
        self.task_name = task_name
        self.sep = sep

        # Ensure task_name is provided if save_emb is enabled
        if save_emb:
            assert self.task_name is not None

        # Create directory for saving embeddings
        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        """
        Encode queries using the specified model engine.

        Args:
        - queries: List of queries to be encoded.
        - batch_size: Batch size for encoding.

        Returns:
        - List of encoded query embeddings.
        """
        fin_embeddings = []

        # Generate a unique filename for the query embeddings
        filename = f"{self.task_name}_{queries[0][:10]}_{queries[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "").replace(":", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        # Check if embeddings are already saved for the queries
        if queries and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            # Encode queries in batches
            for i in range(0, len(queries), batch_size):
                batch = queries[i: i + batch_size]
                response = self.engine.encode(batch)
                fin_embeddings.extend(response)

        # Save embeddings if enabled
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        # Ensure consistency in the number of queries and embeddings
        assert len(queries) == len(fin_embeddings)
        return fin_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        """
        Encode corpus using the specified model engine.

        Args:
        - corpus: List of documents to be encoded.
        - batch_size: Batch size for encoding.

        Returns:
        - List of encoded corpus embeddings.
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
                (doc["title"] + self.sep + doc["text"]
                 ).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        fin_embeddings = []

        # Generate a unique filename for the corpus embeddings
        filename = f"{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        # Check if embeddings are already saved for the corpus
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            # Encode corpus in batches
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i: i + batch_size]
                response = self.engine.encode(batch)
                fin_embeddings.extend(response)

        # Save embeddings if enabled
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        # Ensure consistency in the number of sentences and embeddings
        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()

    # define if only want to run for specific queries in the file
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)

    # define model engine, language, tasks if have specific task and batch size
    parser.add_argument("--engine", type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--queriesfile", type=str, default='queries.jsonl')

    args = parser.parse_args()
    return args

# Main function to execute the retrieval and evaluation process
def main(args):

    # Set up paths and directories
    out_dir = pathlib.Path(__file__).parent.parent.absolute()
    model_name = args.engine.split("/")[-1].split("_")[-1]
    result_path = os.path.join(f"BeIR_Results/{model_name}/")
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

    # Loop through specified tasks and perform retrieval and evaluation
    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)

        # Define paths and filenames
        result_file = task.replace("/", "")
        save_path = f"{result_path}/{result_file}.json"

         # Check if results for the task already exist, if yes, skip the task
        if os.path.exists(save_path):
            logger.warning(
                f"WARNING: {task} results already exists. Skipping.")
        else:

             # Load dataset, queries, and query relevance judgments
            data_path = os.path.join(out_dir, f"Datasets/{task}")

            corpus, queries, qrels = GenericDataLoader(
                data_folder=data_path, 
                query_file=args.queriesfile
                ).load(split="test")

            # Create retrieval model with specified embedder and parameters
            model = DRES(
                Embedder(engine=args.engine, task_name=task, batch_size=args.batchsize, save_emb=True),
                batch_size=args.batchsize,
                corpus_chunk_size=10000)
            retriever = EvaluateRetrieval(model, score_function="dot")
            results = retriever.retrieve(corpus, queries)

            # Perform retrieval and evaluate the results
            tick = time.time()
            ndcg, _map, recall, precision = retriever.evaluate(
                qrels, results, retriever.k_values)
            task_results = {
                "BeIR_version": "2.0.0",
                "BeIR_dataset_name": task,
            }
            mrr = retriever.evaluate_custom(
                qrels, results, retriever.k_values, "mrr")
            tock = time.time()
            scores = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }
            scores["evaluation_time"] = round(tock - tick, 2)
            task_results["test"] = scores

            # Save the evaluation results in a JSON file
            with open(save_path, "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)

# Entry point for the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
