import pathlib
import logging
import sys
import argparse
from rdflib import Graph
from sentence_transformers import SentenceTransformer
# from rdflib import HDTStore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# create embeddings from rdf triples
def main(input_file: pathlib.Path):
    logger.info(f"input: {input_file}")

    # store = HDTStore(input_file)
    # g = Graph(store=store)
    g = Graph()
    triples = []
    try:
        # parse the file
        g.parse(input_file, format="ttl")
        logger.debug("Triples in the file:\n")
        for subj, pred, obj in g:
            logger.debug(f"Subject: {subj}\nPredicate: {pred}\nObject: {obj}\n")
            triple_str = f"{subj} {pred} {obj}"
            triples.append(triple_str)
    except Exception as e:
        logger.error(f"An error occurred while parsing file {input_file}: {e}")

    try:
        # load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")

        # create embeddings
        logger.debug("about to encode embeddings")
        embeddings = model.encode(triples)
        for i in range(min(10, len(triples))):
            logger.debug(f"Triple: {triples[i]}")
            logger.debug(f"Embeddings: {embeddings[i][:10]}...\n")
    except Exception as e:
        logger.error(f"An error occurred while embedding triples {input_file}: {e}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An ht file from an rdf graph')
    args = parser.parse_args()

    main(args.input)
