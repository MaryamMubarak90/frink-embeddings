# frink-embeddings

### Example usage
```
usage: main.py [-h] -i INPUT -c CONF [-q QDRANT_URL] [--csv] [--json]

frink-embeddings

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        An hdt file from an rdf graph
  -c CONF, --conf CONF  The yaml file for configuration
  -q QDRANT_URL, --qdrant_url QDRANT_URL
                        The url for the Qdrant client
  --csv                 Write the output to a csv file
  --json                Write the output to a json file
