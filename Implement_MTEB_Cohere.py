import argparse
import logging

from mteb import MTEB
from Embedders.Cohere_Embedder import CohereEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of tasks for retrieval
TASK_LIST = [
    "ArguAna",
    "ClimateFEVER",
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
    "DBPedia",
    "FEVER",
    "FiQA2018",
    "HotpotQA",
    "MSMARCO",
    "NFCorpus",
    "NQ",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()

    # Define if only want to run for specific queries in the file
    parser.add_argument("--startid", type=int)
    parser.add_argument("--endid", type=int)

    # Define model engine, language, tasks if have specific task and batch size
    parser.add_argument("--engine", type=str, default="embed-english-v3.0")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--taskname", type=str, default=None)
    parser.add_argument("--batchsize", type=int, default=2048)
    parser.add_argument("--API_key", type=str, default="YOUR API KEY")

    args = parser.parse_args()
    return args

def main(args):
    """
    Main function for running the Cohere-based retrieval.
    """
    for task in TASK_LIST[args.startid:args.endid]:
        print("Running task: ", task)

        # Create an instance of CohereEmbedder with specified parameters
        model = CohereEmbedder(args.engine,
                               task_name=task,
                               batch_size=args.batchsize,
                               save_emb=True,
                               API_KEY=args.API_key,
                               base_path=f"MTEB-Embeddings/{args.engine.split('/')[-1]}/")

        # Set evaluation splits based on the task
        eval_splits = ["validation"] if task == "MSMARCO" else ["test"]

        # Extract model name from engine for naming result folder
        model_name = args.engine.split("/")[-1].split("_")[-1]

        # Create an instance of MTEB for evaluation
        evaluation = MTEB(tasks=[task], task_langs=[args.lang])

        # Run the evaluation with specified parameters
        evaluation.run(model,
                       output_folder=f"MTEB_Results/{model_name}",
                       batch_size=args.batchsize,
                       eval_splits=eval_splits,
                       corpus_chunk_size=10000)

if __name__ == "__main__":
    args = parse_args()
    main(args)