import pathlib
import logging
import sys
import os
import csv
import argparse
import yaml
from rdflib import URIRef
from rdflib_hdt import HDTDocument
from sentence_transformers import SentenceTransformer

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
    iri_types = []

    try:
        with open(conf_file, 'r') as conf:
            conf_yaml = yaml.safe_load(conf)
            iri_types = conf_yaml['iriTypes']

        conf.close()

    except Exception as e:
        logger.error(f"An error occurred while parsing file {conf_file}: {e}")

    return iri_types

# function to determine whether this IRI should
# be included in list of triple to be embedded
# this function consults a config file
# to determine which IRI types to include
def needsEmbedding(iri_types, predicate):
    ret = False

    if str(predicate) in iri_types:
        ret = True

    return ret

# create embeddings from rdf triples
def main(input_file: pathlib.Path, config_file: pathlib.Path):
    logger.info(f"input: {input_file}  config: {config_file}")

    doc = HDTDocument(str(input_file))
    logger.info(f"subjects: {doc.nb_subjects}  predicates: {doc.nb_predicates}  objects: {doc.nb_objects}")

    iri_types = getIriTypes(config_file)

    subject = None
    object_list = []
    triples_dict = {}
    triples_iterator, cardinality = doc.search((None, None, None))
    # loop through triples that are IRIs
    # since triple subjects seem to be in order
    # (ignore relationships for now?)
    # also, only include IRI predicates that are
    # listed in the config file
    try:
        for s, p, o in triples_iterator:
            if isinstance(p, URIRef):

                if subject is None: # first time through
                    subject = str(s)
                    # check config file here to make sure we
                    # want to include this predicate in the embedding
                    if needsEmbedding(iri_types, p):
                        object_list.append(f"{str(p)} {str(o)}")

                elif str(s) != subject: # we have changed to a new subject
                    # save subject and object list from last interation
                    if len(object_list) > 0:
                        triples_dict[subject] = ";".join(object_list)
                        # reset object list
                        object_list = []
                    # new subject
                    subject = str(s)
                    if needsEmbedding(iri_types, p):
                        object_list.append(f"{str(p)} {str(o)}")

                else: # still on same subject, collecting predicate/object pairs
                    # check config file here to make sure we
                    # want to include this predicate in the embedding
                    if needsEmbedding(iri_types, p):
                        object_list.append(f"{str(p)} {str(o)}")

    except Exception as e:
        logger.error(f"An error occurred while parsing file {input_file}: {e}")
    try:
        # load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")

        # create embeddings
        logger.debug("about to encode embeddings")
        # create an embedding for each unique subject
        # and write to the csv file
        csv_file = os.path.splitext(os.path.basename(input_file))[0] + ".csv"
        with open(csv_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['iri', 'embedding'])
            for k, v in triples_dict.items():
                embedding = model.encode(v)
                writer.writerow([k, embedding])
        f.close()

    except Exception as e:
        logger.error(f"An error occurred while embedding triples {input_file}: {e}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    args = parser.parse_args()

    main(args.input, args.conf)
