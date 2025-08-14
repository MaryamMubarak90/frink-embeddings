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
from qdrant_client import QdrantClient, models

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

def getIriKeyFromValue(iri_types, iri_type_value):

    for iris in iri_types:
        for key, value in iris.items():
            if isinstance(value, list) and iri_type_value in value:
                return key

    return None


# get list of iri types from config file
def getIriTypes(conf_file):
    iri_types = []

    try:
        with open(conf_file, 'r') as conf:
            conf_yaml = yaml.safe_load(conf)
            iri_types = conf_yaml['irisToEmbed']

        conf.close()

    except Exception as e:
        logger.error(f"An error occurred while parsing file {conf_file}: {e}")

    return iri_types

def getAllIriTypeValues(iri_types):
    iri_values = []

    for type in iri_types:
        value_list_obj = type.values()
        value_list = list(value_list_obj)[0]
        iri_values += value_list

    return iri_values


# this function takes a list of dicts and creates
# another dict that take the format of:
# <triple subject>: <string - sentence to embed>
# the string to embed looks something like this:
# 'HELLO; also known as  ERS-1_BYU_L3_OW_SIGMA0_ENHANCED; description: WHEEEEE!; subject: HELLO; subject: ERS-1 Gridded Level 3 Enhanced Resolution Sigma-0 from BYU'
# note the labels are a special case and not preceded by a <key>: before the value
def createSentences(embed_list):

    sentence_dict = {}
    sentences_to_embed_dict = {}

    for item in embed_list:
        subject = item.get('subject')
        obj = item.get('object')
        key = item.get('config_key')
        if subject is not None and key is not None:
            if subject not in sentence_dict:
                sentence_dict[subject] = []
            sentence_dict[subject].append({key: obj})

    # now create sentence to embed for each subject (s)
    also_str = "also known as"
    for key, value in sentence_dict.items():
        sentence = None

        # special case - don't add the key for labels
        # use following for testing
        # val = [{'label': 'HELLO'},{'description': 'WHEEEEE!'},{'subject': 'HELLO'},{'subject': 'ERS-1 Gridded Level 3 Enhanced Resolution Sigma-0 from BYU'},{'label': 'ERS-1_BYU_L3_OW_SIGMA0_ENHANCED'}]
        labels = [d["label"] for d in value if "label" in d]
        if labels is not None:
            sentence = labels[0]
            if len(labels) > 1:
                sentence += f"; {also_str} "
                for label in labels[1:]:
                    sentence += f" {label}"

        # now append the rest of the predicates
        for value_dict in value:
            for k, v in value_dict.items():
                if k != "label":
                    sentence += f"; {k}: {v}"

        sentences_to_embed_dict[key] = sentence

    return sentences_to_embed_dict


# create embeddings from rdf triples
def main(input_file: pathlib.Path, config_file: pathlib.Path):
    logger.info(f"input: {input_file}  config: {config_file}")

    doc = HDTDocument(str(input_file))
    logger.info(f"subjects: {doc.nb_subjects}  predicates: {doc.nb_predicates}  objects: {doc.nb_objects}")

    iri_types = getIriTypes(config_file)
    iri_values = getAllIriTypeValues(iri_types)

    sentences_to_embed_dict = {}
    embed_list = []
    triples, cardinality = doc.search((None, None, None))

    try:
        # first collect all of the triples desired - as defined
        # by the predicates listed in the config file
        for s, p, o in triples:
            if str(p) in iri_values and isinstance(p, URIRef):
                key = getIriKeyFromValue(iri_types, str(p))
                embed_dict = {"config_key": key, "subject": str(s), "predicate": str(p), "object": str(o)}
                embed_list.append(embed_dict)

        # now create a list of sentences to embed
        sentences_to_embed_dict = createSentences(embed_list)


    except Exception as e:
        logger.error(f"An error occurred while parsing file {input_file}: {e}")
    try:
        # load embedding model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")

        # create embeddings and write to a csv file
        # logger.debug("about to encode embeddings")
        # create an embedding for each unique subject
        # and write to the csv file
        # csv_file = os.path.splitext(os.path.basename(input_file))[0] + ".csv"
        # with open(csv_file, "w", newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(['iri', 'embedding'])
        #     for k, v in sentences_to_embed_dict.items():
        #         embedding = model.encode(v)
        #         writer.writerow([k, embedding])
        # f.close()

        logger.debug("about to encode embeddings")

        # create an embedding for each unique subject
        # and write to a qdrant collection
        # now connect to Qdrant client
        client = QdrantClient(url="http://localhost:6333", timeout=30)
        # create a collection name
        collection_name = os.path.splitext(os.path.basename(input_file))[0]
        if client.collection_exists(collection_name):
            # collection already exists so notify and exit
            logger.info(f"Collection '{collection_name}' already exists.")
            exit(0)
        else:
            # collection does not exist - continue to create and populate
            logger.debug(f"Collection '{collection_name}' does not exist.")
            client.create_collection(collection_name=collection_name,
                                     vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE))
            idx = 0
            points = []
            print(len(sentences_to_embed_dict.items()))
            for k, v in sentences_to_embed_dict.items():
                idx+=1
                embedding = model.encode(v)
                points.append(
                    models.PointStruct(
                        id=idx,  # Unique ID for each point
                        vector=embedding,
                        payload={"iri": k}  # Add metadata (payload)
                    )
                )
                # batch upserts to avoid timeouts
                if idx % 1000 == 0:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []

            # final upsert
            client.upsert(collection_name=collection_name, points=points)

    except Exception as e:
        logger.error(f"An error occurred while embedding triples {input_file}: {e}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    args = parser.parse_args()

    main(args.input, args.conf)
