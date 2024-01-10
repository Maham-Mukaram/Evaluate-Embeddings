import argparse
import json
import logging
import os
import pathlib
import pickle
import time
from typing import List, Dict

from transformers import AutoTokenizer

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from sentence_transformers import SentenceTransformer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/w/340/mahamm/.cache/Models"
os.environ["HF_DATASETS_CACHE"]="/w/340/mahamm/.cache/Datasets"
os.environ["HF_MODULES_CACHE"]="/w/340/mahamm/.cache/Modules"
os.environ["HF_METRICS_CACHE"]="/w/340/mahamm/.cache/Metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY = 

TASK_LIST = [
    "scifact","trec-covid","webis-touche2020"
]

class bgeEmbedder:
    """
    Benchmark Cohere embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=64, sep=" ", save_emb=False, **kwargs):
        self.engine = SentenceTransformer(engine)
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.base_path = f"BeIR-Embeddings/{engine.split('/')[-1]}/"
        self.task_name = task_name
        self.sep = sep

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
    
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):

        fin_embeddings = []

        filename = f"CoT_{self.task_name}_{queries[0][:10]}_{queries[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "").replace(":", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        if queries and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i : i + self.batch_size]
                response = self.engine.encode(batch)
                # May want to sleep here to avoid getting too many requests error
                fin_embeddings.extend(response)

        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(queries) == len(fin_embeddings)
        return fin_embeddings

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip()
                for doc in corpus
            ]

        fin_embeddings = []

        filename = f"{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]
                response = self.engine.encode(batch)
                # May want to sleep here to avoid getting too many requests error
                fin_embeddings.extend(response)

        # Save embeddings
        if fin_embeddings and self.save_emb:
            dump = {
                "fin_embeddings": fin_embeddings,
            }
            pickle.dump(dump, open(embedding_path, "wb"))

        assert len(sentences) == len(fin_embeddings)
        return fin_embeddings



def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)
    parser.add_argument("--engine", type=str, default='BAAI/bge-base-en-v1.5')
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args

def main(args):

    # There are two different batch sizes
    # VoyageEmbedder
    #(...) batch size arg is used to send X embeddings to the API
    # evaluation.run(...) batch size arg is how much will be saved / pickle file (as it's the total sent to the embed function)
    out_dir = pathlib.Path(__file__).parent.parent.absolute()
    model_name = args.engine.split("/")[-1].split("_")[-1]
    result_path = os.path.join(f"BeIR_Results/{model_name}/CoT_2")
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        result_file = task.replace("/", "")
        save_path= f"{result_path}/{result_file}.json"
        if os.path.exists(save_path):
            logger.warning(f"WARNING: {task} results already exists. Skipping.")
        else:
            data_path  = os.path.join(out_dir, f"Datasets/BeIR/{task}")
            eval_splits = "validation" if task == "MSMARCO" else "test"
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path, query_file='expanded_queries/queries_CoT.jsonl').load(split=eval_splits)
            model = DRES(bgeEmbedder
        (engine=args.engine, task_name=task, batch_size=args.batchsize, save_emb=True),batch_size=args.batchsize, corpus_chunk_size=10000)
            retriever = EvaluateRetrieval(model, score_function="dot")
            results = retriever.retrieve(corpus, queries)
            tick = time.time()
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            task_results = {
                        "BeIR_version": "2.0.0",
                        "BeIR_dataset_name": task,
                    }
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, "mrr")
            tock = time.time()

            scores = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }
            scores["evaluation_time"] = round(tock - tick, 2)
            task_results[eval_splits] = scores
            with open(save_path, "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)

if __name__ == "__main__":
    args = parse_args()
    main(args)



    