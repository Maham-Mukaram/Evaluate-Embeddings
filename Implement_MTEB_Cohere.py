import argparse
import logging
import os
import pathlib
import pickle
import time
from typing import List, Dict

from mteb import MTEB
import cohere
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="/w/339/mahamm/Models"
os.environ["HF_DATASETS_CACHE"]="/w/339/mahamm/Datasets"
os.environ["HF_MODULES_CACHE"]="/w/339/mahamm/Modules"
os.environ["HF_METRICS_CACHE"]="/w/339/mahamm/Metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

API_KEY = "tzyquyok6KIAh8gsEsfLzapvexvj3jYFDbsl4eHO"

TASK_LIST_CLASSIFICATION = [
]

TASK_LIST_CLUSTERING = [
]

TASK_LIST_PAIR_CLASSIFICATION = [
]

TASK_LIST_RERANKING = [
]

TASK_LIST_RETRIEVAL = [
    "ArguAna",
    #"ClimateFEVER",
    "CQADupstackAndroidRetrieval",
    "CQADupstackEnglishRetrieval",
    "CQADupstackGamingRetrieval",
    "CQADupstackGisRetrieval",
    "CQADupstackMathematicaRetrieval",
    "CQADupstackPhysicsRetrieval",
    "CQADupstackProgrammersRetrieval",
    "CQADupstackStatsRetrieval",
    "CQADupstackTexRetrieval",
    "CQADupstackUnixRetrieval",
    "CQADupstackWebmastersRetrieval",
    "CQADupstackWordpressRetrieval",
    #"DBPedia",
    #"FEVER",
    "FiQA2018",
    #"HotpotQA",
    #"MSMARCO",
    "NFCorpus",
    #"NQ",
    #"QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    #"Touche2020",
    #"TRECCOVID",
]

# TASK_LIST_STS = [
#      TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_STS
# ]

TASK_LIST =TASK_LIST_RETRIEVAL 

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

    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        model = CohereEmbedder(args.engine, task_name=task, batch_size=args.batchsize, save_emb=True)
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]
        model_name = args.engine.split("/")[-1].split("_")[-1]
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])
        evaluation.run(model, output_folder=f"results-cohere/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits, corpus_chunk_size=10000)

if __name__ == "__main__":
    args = parse_args()
    main(args)



    