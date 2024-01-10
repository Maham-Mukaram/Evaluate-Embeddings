"""
openai==0.26.4
tiktoken==0.2.0
"""
import argparse
import logging
import os
import pathlib
import pickle
import time

from mteb import MTEB
import cohere
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)

os.environ["HF_DATASETS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_OFFLINE"]="0" # 1 for offline
os.environ["TRANSFORMERS_CACHE"]="commun/models"
os.environ["HF_DATASETS_CACHE"]="commun/datasets"
os.environ["HF_MODULES_CACHE"]="commun/modules"
os.environ["HF_METRICS_CACHE"]="commun/metrics"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK_LIST_CLASSIFICATION = [
    # "AmazonCounterfactualClassification",
    # "AmazonPolarityClassification",
    # "AmazonReviewsClassification",
    # "Banking77Classification",
    # "EmotionClassification",
    # "ImdbClassification",
    # "MassiveIntentClassification",
    # "MassiveScenarioClassification",
    # "MTOPDomainClassification",
    # "MTOPIntentClassification",
    # "ToxicConversationsClassification",
    # "TweetSentimentExtractionClassification",
]

TASK_LIST_CLUSTERING = [
    # "ArxivClusteringP2P",
    # "ArxivClusteringS2S",
    # "BiorxivClusteringP2P",
    # "BiorxivClusteringS2S",
    # "MedrxivClusteringP2P",
    # "MedrxivClusteringS2S",
    # "RedditClustering",
    # "RedditClusteringP2P",
    # "StackExchangeClustering",
    # "StackExchangeClusteringP2P",
    # "TwentyNewsgroupsClustering",
]

TASK_LIST_PAIR_CLASSIFICATION = [
    # "SprintDuplicateQuestions",
    # "TwitterSemEval2015",
    # "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    # "AskUbuntuDupQuestions",
    # "MindSmallReranking",
    # "SciDocsRR",
    # "StackOverflowDupQuestions",
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

TASK_LIST_STS = [
    # "BIOSSES",
    # "SICK-R",
    # "STS12",
    # "STS13",
    # "STS14",
    # "STS15",
    # "STS16",
    # "STS17",
    # "STS22",
    # "STSBenchmark",
    # "SummEval",
]

TASK_LIST = TASK_LIST_CLASSIFICATION + TASK_LIST_CLUSTERING + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_RERANKING + TASK_LIST_RETRIEVAL + TASK_LIST_STS

class CohereEmbedder:
    """
    Benchmark Cohere embeddings endpoint.
    """
    def __init__(self, engine, task_name=None, batch_size=32, save_emb=False, API_KEY="V", **kwargs):
        self.engine = engine
        self.max_token_len = 8191
        self.batch_size = batch_size
        self.save_emb = save_emb # Problematic as the filenames may end up being the same
        self.base_path = f"test/embeddings-cohere/{engine.split('/')[-1]}/"
        self.tokenizer = AutoTokenizer.from_pretrained("Cohere/Cohere-embed-english-v3.0")
        self.task_name = task_name
        self.API_KEY=API_KEY

        if save_emb:
            assert self.task_name is not None

        pathlib.Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
    def encode(self, 
            sentences,
            decode=True,
            idx=None,
            **kwargs
        ):
        co = cohere.Client(self.API_KEY)

        fin_embeddings = []

        embedding_path = f"{self.base_path}/{self.task_name}_{sentences[0][:10]}_{sentences[-1][-10:]}.pickle"
        if sentences and os.path.exists(embedding_path):
            loaded = pickle.load(open(embedding_path, "rb"))
            fin_embeddings = loaded["fin_embeddings"]
        else:
            call_count=0
            for i in range(0, len(sentences), self.batch_size):
                batch = sentences[i : i + self.batch_size]

                all_tokens = []
                used_indices = []
                for j, txt in enumerate(batch):
                    # tokens = self.tokenizer.encode(txt, add_special_tokens=False)
                    if not(txt):
                        print("Detected empty item, which is not allowed by the Cohere API - Replacing with empty space")
                        txt = " "
                    tokens = self.tokenizer.encode(txt)
                    token_len = len(tokens)
                    if token_len > self.max_token_len:
                        tokens = tokens[:self.max_token_len]
                    # For some characters the API raises weird errors, e.g. input=[[126]]
                    if decode:
                        tokens = self.tokenizer.decode(tokens)
                    all_tokens.append(tokens)
                    used_indices.append(j)

                if all_tokens:
                    response = co.embed(all_tokens, input_type="search_document", model=self.engine).embeddings
                    call_count = call_count+1
                    # May want to sleep here to avoid getting too many requests error
                    time.sleep(15)
                    assert len(response) == len(
                        all_tokens
                    ), f"Sent {len(all_tokens)}, got {len(response)}"

                fin_embeddings.extend(response)
            print(call_count)
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
    parser.add_argument("--API_KEY", type=int, default=2048)
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
        evaluation.run(model, output_folder=f"test/results-cohere/{model_name}", batch_size=args.batchsize, eval_splits=eval_splits, corpus_chunk_size=10000, API_KEY=args.API_KEY)

if __name__ == "__main__":
    args = parse_args()
    main(args)



    