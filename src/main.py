import pathlib
import logging
import sys
import argparse
import yaml
# from rdflib import Graph
#from rdflib_hdt import HDTStore
from rdflib import URIRef
from rdflib_hdt import HDTDocument
from sentence_transformers import SentenceTransformer


triples = []

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
# get list of iri types from config file
def getIriTypes(conf_file):
    iri_types = ''
    try:
        with open(conf_file, 'r') as conf:
            conf_yaml = yaml.safe_load(conf)
            iri_types = conf_yaml['iriTypes']
    except Exception as e:
        logger.error(f"An error occurred while parsing file {conf_file}: {e}")

    conf.close()
    return iri_types

# function to determine whether this IRI should
# be included in list of triple to be embedded
# this function consults a config file
# to determine which IRI types to include
def needsEmbedding(iri_types, predicate):
    ret = False

    if predicate in iri_types:
        ret = True

    return ret

# create embeddings from rdf triples
def main(input_file: pathlib.Path, config_file: pathlib.Path):
    logger.info(f"input: {input_file}  config: {config_file}")

    doc = HDTDocument(str(input_file))
    #s = HDTStore(str(input_file))
    #g = Graph(store=s)
    iri_types = getIriTypes(config_file)
    triples = []
    triples_iterator, cardinality = doc.search((None, None, None))
    try:
        for s, p, o in triples_iterator:
            # The 'p' in (s, p, o) represents the predicate
            if isinstance(p, URIRef) and needsEmbedding(iri_types, p):
                triple_str = f"{s} {p} {o}"
                triples.append(triple_str)
        # parse the file this example using a Graph
        #g.parse(input_file, format="hdt")
        #logger.debug("Triples in the file:\n")
        #for subj, pred, obj in g:
            #logger.debug(f"Subject: {subj}\nPredicate: {pred}\nObject: {obj}\n")
            #triple_str = f"{subj} {pred} {obj}"
            #triples.append(triple_str)
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
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    args = parser.parse_args()

    main(args.input, args.conf)
