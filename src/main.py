#How to Run

# Original behavior (no types): python main.py -i graph_name.hdt -c config.yaml --json

# For new behavior (Type-aware): python main.py -i graph_name.hdt -c config.yaml --json \ --type-aware --types-json subject_types.json --max-types 3

# "subject_types.json" file represents a mapping from each subject IRI towards RDF graph to the list of type IRIs that describe what that subject “is an instance of.” It is used only when we enable --type-aware, so the script can append type information in natural language form.


import pathlib
import logging
import sys
import os
import csv
import json
import numpy as np
import argparse
import yaml
from typing import Dict, List, Optional, Iterable

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

def iri_tail(iri: str) -> str:
    """Return a readable tail from an IRI/URI for fallback labeling."""
    if not iri:
        return ""
    tail = iri.rsplit("/", 1)[-1]
    tail = tail.rsplit("#", 1)[-1]
    return tail

def prettify_from_iri(iri: str) -> str:
    """
    Derive a human-ish label from an IRI tail.
    Example: 'StainlessSteelProcessingCapability' -> 'Stainless Steel Processing Capability'
             'HP_0004321' -> 'HP 0004321'
    """
    tail = iri_tail(iri)
    if not tail:
        return iri or ""

    out = []
    token = ""
    for ch in tail:
        if ch in {"_", "-"}:
            if token:
                out.append(token)
                token = ""
        elif ch.isupper() and token and not token[-1].isupper():
            out.append(token)
            token = ch
        else:
            token += ch
    if token:
        out.append(token)

    label = " ".join(out).strip()
    return label or iri

def normalize_label(s: Optional[str]) -> str:
    return (s or "").strip()

def english_with_types(
    label_text: str,
    type_terms: Iterable[str],
    max_types: Optional[int] = None
) -> str:
    """
    Single, English-style phrasing for type-aware context (per review):
      "<label>, an instance of <type>"
      "<label>, an instance of <type1>; <type2>; ..."
    If a type looks like an IRI, render a readable label from its tail.
    """
    label_text = normalize_label(label_text)
    terms = list(type_terms or [])
    if not terms:
        return label_text

    if max_types is not None and max_types >= 0:
        terms = terms[:max_types]

    rendered: List[str] = []
    for t in terms:
        t = (t or "").strip()
        if not t:
            continue
        if t.startswith("http://") or t.startswith("https://"):
            rendered.append(prettify_from_iri(t))
        else:
            rendered.append(t)

    if not rendered:
        return label_text

    if len(rendered) == 1:
        return f"{label_text}, an instance of {rendered[0]}"
    return f"{label_text}, an instance of " + "; ".join(rendered)

# create embeddings and write to a tsv file
def saveToTSV(model, input_file, sentences_to_embed_dict):
    tsv_file = ""
    embedding_list = []
    try:
        tsv_file = os.path.splitext(os.path.basename(input_file))[0] + ".tsv"
        with open(tsv_file, "w", newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['iri', 'label', 'embedding'])
            for k, v in sentences_to_embed_dict.items():
                embedding = model.encode(v)
                writer.writerow([k, v, np.array2string(embedding, separator=', ').replace('\n', '')])
        f.close()
        logger.info(f"Saved embeddings to {tsv_file}")
    except Exception as e:
        logger.error(f"An error occurred while save embedded data to the tsv file:{tsv_file} {e}")

# create embeddings and write to a json file suitable for uploading into a vector database
def saveToJSON(model, input_file, sentences_to_embed_dict):
    graph_name = os.path.splitext(os.path.basename(input_file))[0]
    json_file = os.path.splitext(os.path.basename(input_file))[0] + ".json"
    dict_list = []
    idx = 1
    try:
        for k, v in sentences_to_embed_dict.items():
            embedding = model.encode(v)
            embedded_dict = {
                "id": idx,
                "vector": embedding.tolist(),
                "payload": {"graph": graph_name, "iri": k, "label": v}
            }
            dict_list.append(embedded_dict)
            idx += 1

        with open(json_file, 'w') as f:
            json.dump({'points': dict_list}, f, indent=3)

        f.close()
        logger.info(f"Saved embeddings to {json_file}")
    except Exception as e:
        logger.error(f"An error occurred while saving embedded data to the json file:{json_file} {e}")

# create an embedding for each unique subject and write to a qdrant collection
def saveToQdrant(model, url, input_file, sentences_to_embed_dict):
    # connect to Qdrant client
    try:
        client = QdrantClient(url=url, timeout=30)
        # create a collection name
        collection_name = os.path.splitext(os.path.basename(input_file))[0]
        if client.collection_exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            exit(0)
        else:
            logger.debug(f"Collection '{collection_name}' does not exist.")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
            )
            idx = 0
            points = []

            for k, v in sentences_to_embed_dict.items():
                idx += 1
                embedding = model.encode(v)
                points.append(
                    models.PointStruct(
                        id=idx,
                        vector=embedding,
                        payload={"graph": collection_name, "iri": k, "label": v}
                    )
                )
                if idx % 1000 == 0:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []

            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Saved embeddings to Qdrant collection '{collection_name}'.")

    except Exception as e:
        logger.error(f"An error occurred uploading data to Qdrant:{url}: {e}")

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
    for t in iri_types:
        value_list = list(t.values())[0]
        iri_values += value_list
    return iri_values

# counts how many strings in the sentence list match the target string
def countDuplicates(string_list, target):
    return sum(1 for s in string_list if s.lower() == target.lower())

def createSentences(
    embed_list: List[Dict[str, str]],
    *,
    type_aware: bool = False,
    types_by_subject: Optional[Dict[str, List[str]]] = None,
    max_types: Optional[int] = None
) -> Dict[str, str]:
    """
    Build a dict: subject IRI -> text to embed.

    Base (original) format:
      "<label>; also known as <alias1> <alias2> ...; description: ...; subject: ...; ..."

    Type-aware extension (per review) — applied exactly once here:
      If --type-aware and types exist for subject:
        "<label>, an instance of <type1>; <type2>; ...; also known as ...; description: ...; ..."

    If no label is present, we fall back to a prettified label from the subject IRI tail.
    """
    sentence_dict: Dict[str, List[Dict[str, str]]] = {}
    sentences_to_embed_dict: Dict[str, str] = {}
    types_by_subject = types_by_subject or {}

    # Collect by subject
    for item in embed_list:
        subject = item.get('subject')
        obj = item.get('object')
        key = item.get('config_key')  # subject IRI (same as `subject`, but preserving your naming)
        if subject is not None and key is not None:
            if subject not in sentence_dict:
                sentence_dict[subject] = []
            sentence_dict[subject].append({key: obj})

    also_str = "also known as"

    # Build sentence per subject
    for subj, kv_list in sentence_dict.items():
        # Gather labels first
        labels = [d["label"] for d in kv_list if "label" in d]
        label_main = normalize_label(labels[0]) if (labels and labels[0]) else prettify_from_iri(subj)

        sentence = label_main

        # Append types (English phrasing) before other predicates
        if type_aware:
            type_terms = types_by_subject.get(subj, [])
            sentence = english_with_types(
                label_text=sentence,
                type_terms=type_terms,
                max_types=max_types
            )

        # Handle additional labels as "also known as"
        if labels and len(labels) > 1:
            # only add "also known as" if not all duplicates of the main label
            if countDuplicates(labels, label_main) != len(labels):
                sentence += f"; {also_str}"
                for label in labels[1:]:
                    lab = normalize_label(label)
                    if lab and lab not in sentence:
                        sentence += f" {lab}"

        # Append the rest of the predicates (non-label keys)
        for value_dict in kv_list:
            for k, v in value_dict.items():
                if k != "label":
                    sentence += f"; {k}: {v}"

        sentences_to_embed_dict[subj] = sentence

    return sentences_to_embed_dict


def main(input_file: pathlib.Path, config_file: pathlib.Path, tsv_output, json_output, qdrant_url,
         type_aware: bool = False, types_json: Optional[pathlib.Path] = None, max_types: Optional[int] = None):
    logger.info(f"input: {input_file}  config: {config_file}")

    doc = HDTDocument(str(input_file))
    logger.info(f"subjects: {doc.nb_subjects}  predicates: {doc.nb_predicates}  objects: {doc.nb_objects}")

    iri_types = getIriTypes(config_file)
    iri_values = getAllIriTypeValues(iri_types)

    embed_list: List[Dict[str, str]] = []
    sentences_to_embed_dict: Dict[str, str] = {}
    model = None
    triples, cardinality = doc.search((None, None, None))

    # Optional types map (subject IRI -> [types]) for type-aware phrasing
    types_by_subject: Optional[Dict[str, List[str]]] = None
    if types_json is not None and types_json.exists():
        try:
            types_by_subject = json.loads(types_json.read_text(encoding="utf-8"))
            logger.info(f"Loaded types map from {types_json} with {len(types_by_subject)} subjects.")
        except Exception as e:
            logger.error(f"Failed to read types JSON {types_json}: {e}")
            types_by_subject = None

    try:
        # Collect triples according to config predicates
        for s, p, o in triples:
            if str(p) in iri_values and isinstance(p, URIRef):
                key = getIriKeyFromValue(iri_types, str(p))
                embed_dict = {"config_key": key, "subject": str(s), "predicate": str(p), "object": str(o)}
                embed_list.append(embed_dict)

        # Build sentences (single point where type-aware is applied)
        sentences_to_embed_dict = createSentences(
            embed_list,
            type_aware=type_aware,
            types_by_subject=types_by_subject,
            max_types=max_types
        )

    except Exception as e:
        logger.error(f"An error occurred while parsing file {input_file}: {e}")

    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.debug("loaded model")
    except Exception as e:
        logger.error(f"An error occurred while loading the embedding model: {e}")

    # Outputs (original flags preserved)
    if tsv_output:
        saveToTSV(model, input_file, sentences_to_embed_dict)

    if json_output:
        saveToJSON(model, input_file, sentences_to_embed_dict)

    if qdrant_url is not None:
        saveToQdrant(model, qdrant_url, input_file, sentences_to_embed_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='frink-embeddings')
    parser.add_argument('-i', '--input', required=True, type=pathlib.Path, help='An hdt file from an rdf graph')
    parser.add_argument('-c', '--conf', required=True, type=pathlib.Path, help='The yaml file for configuration')
    parser.add_argument('-q', '--qdrant_url', required=False, help='The url for the Qdrant client')
    parser.add_argument('--tsv', action='store_const', const=True, help='Write the output to a tsv file')
    parser.add_argument('--json', action='store_const', const=True, help='Write the output to a json file')

    # New, minimal flags for type-aware behavior (per review)
    parser.add_argument('--type-aware', action='store_const', const=True,
                        help='Include type context in the English sentence (no mode; single phrasing).')
    parser.add_argument('--types-json', type=pathlib.Path, required=False,
                        help='Optional JSON file: { subject_iri: [type1, type2, ...], ... }')
    parser.add_argument('--max-types', type=int, required=False, default=None,
                        help='Optional cap on number of types to append.')

    args = parser.parse_args()
    if not (args.tsv or args.json or args.qdrant_url):
        parser.error("At least one of --tsv, --json or --qdrant_url is required.")

    main(
        args.input,
        args.conf,
        args.tsv,
        args.json,
        args.qdrant_url,
        type_aware=bool(args.type_aware),
        types_json=args.types_json,
        max_types=args.max_types
    )