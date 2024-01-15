import argparse
import json
import logging
import os
import pathlib
import time

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

from Embedders.Cohere_Embedder import CohereEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of tasks to run
TASK_LIST = [
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
    "cqadupstack/wordpress"
]

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    
    # define if only want to run for specific queries in the file
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)

    # define model engine, language, tasks if have specific task and batch size
    parser.add_argument("--engine", type=str, default="embed-english-v3.0")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--queriesfile", type=str, default='queries.jsonl')

    args = parser.parse_args()
    return args

# Main function
def main(args):
    # Get the absolute path of the directory containing the script
    out_dir = pathlib.Path(__file__).parent.parent.absolute()
    model_name = args.engine.split("/")[-1].split("_")[-1]
    result_path = os.path.join(f"BeIR_Results/{model_name}")
    pathlib.Path(result_path).mkdir(parents=True, exist_ok=True)

    # Iterate through tasks in the specified range
    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)
        result_file = task.replace("/", "")
        save_path = f"{result_path}/{result_file}.json"

        # Check if results for the task already exist
        if os.path.exists(save_path):
            logger.warning(f"WARNING: {task} results already exist. Skipping.")
        else:
            # Load dataset for the task
            data_path = os.path.join(out_dir, f"Datasets/{task}")
            eval_splits = "test"
            corpus, queries, qrels = GenericDataLoader(
                data_folder=data_path,
                query_file=args.queriesfile
                ).load(split=eval_splits)

            # Create DenseRetrievalExactSearch model with CohereEmbedder
            model = DRES(
                CohereEmbedder(engine=args.engine, 
                               task_name=task, 
                               batch_size=args.batchsize, 
                               save_emb=True,
                               API_KEY=args.API_key,
                               base_path= f"BeIR-Embeddings/{args.engine.split('/')[-1]}/"),
                batch_size=args.batchsize,
                corpus_chunk_size=10000)

            # Evaluate the retrieval model
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

            # Store evaluation scores in a dictionary
            scores = {
                **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
                **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
                **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
                **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
                **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            }
            scores["evaluation_time"] = round(tock - tick, 2)
            task_results[eval_splits] = scores

            # Save the results to a JSON file
            with open(save_path, "w") as f_out:
                json.dump(task_results, f_out, indent=2, sort_keys=True)

# Entry point of the script
if __name__ == "__main__":
    args = parse_args()
    main(args)
