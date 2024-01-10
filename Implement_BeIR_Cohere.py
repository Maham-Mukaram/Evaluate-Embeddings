import argparse
import json
import logging
import os
import pathlib
import pickle
import time
from typing import List, Dict

from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, WordEmbeddings
from transformers import AutoTokenizer

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import cohere


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/w/339/mahamm/Models"
os.environ["HF_DATASETS_CACHE"]="/w/339/mahamm/Datasets"
os.environ["HF_MODULES_CACHE"]="/w/339/mahamm/Modules"
os.environ["HF_METRICS_CACHE"]="/w/339/mahamm/Metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY = "r5pLYI1ZZ3HGNOswM7wl12p05qEEOXr77VmyKxL0"

TASK_LIST = [
    "arguana",
    #"climate-fever",
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
    #"dbpedia-entity",
    #"fever",
    "fiqa",
    #"germanquad",
    #"hotpotqa",
    #"msmarco",
    "nfcorpus",
    #"nq",
    "quora",
    "scidocs",
    "scifact",
    "webis-touche2020",
    "trec-covid",
]

class CohereEmbedder:
    """
    Benchmark Cohere embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=32, sep=" ", save_emb=False, **kwargs):
        self.engine = engine
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.base_path = f"BeIR-Embeddings/{engine.split('/')[-1]}/"
        self.task_name = task_name
        self.sep = sep

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
    
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):

        co = cohere.Client(API_KEY)

        fin_embeddings = []

        filename = f"{self.task_name}_{queries[0][:10]}_{queries[-1][-10:]}"
        filename = filename.replace("/", "").replace("\\", "")
        embedding_path = f"{self.base_path}/{filename}.pickle"

        if queries and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            for i in range(0, len(queries), self.batch_size):
                batch = queries[i : i + self.batch_size]
                response = co.embed(batch, input_type="search_query", model=self.engine).embeddings
                # May want to sleep here to avoid getting too many requests error
                time.sleep(20)

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
        
        co = cohere.Client(API_KEY)

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
                response = co.embed(batch, input_type="search_document", model=self.engine).embeddings
                # May want to sleep here to avoid getting too many requests error
                time.sleep(20)

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
    parser.add_argument("--engine", type=str, default="embed-english-v3.0")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    args = parser.parse_args()
    return args

def main(args):

    # There are two different batch sizes
    # CohereEmbedder(...) batch size arg is used to send X embeddings to the API
    # evaluation.run(...) batch size arg is how much will be saved / pickle file (as it's the total sent to the embed function)
    out_dir = pathlib.Path(__file__).parent.parent.absolute()
    model_name = args.engine.split("/")[-1].split("_")[-1]
    result_path = os.path.join(f"BeIR_Results/{model_name}")
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
            corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=eval_splits)
            model = DRES(CohereEmbedder(engine=args.engine, task_name=task, batch_size=args.batchsize, save_emb=True),batch_size=args.batchsize, corpus_chunk_size=10000)
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



    