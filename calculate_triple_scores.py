import os
import re
import time
import argparse
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import requests
import spacy
from SPARQLWrapper import SPARQLWrapper, JSON

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


def fuzzy_entity_match(detected_entities, entity, threshold=80):
    """Perform fuzzy matching of an entity against detected entities."""
    for detected_entity in detected_entities:
        if fuzz.partial_ratio(detected_entity, entity) >= threshold:
            return True
    return False


def verify_entities_in_evidence_sentences(sentences, head_entity, tail_entity, evidence_sentence_ids):
    """Verify the presence of entities in evidence sentences."""
    for sentence_data in sentences:
        if sentence_data['Sentence ID'] in evidence_sentence_ids:
            doc = nlp(sentence_data['Sentence'])
            detected_entities = [ent.text for ent in doc.ents]
            
            head_entity_match = fuzzy_entity_match(detected_entities, head_entity)
            tail_entity_match = fuzzy_entity_match(detected_entities, tail_entity)
            
            if head_entity_match and tail_entity_match:
                return True
    return False


def query_falcon_api(kb, text):
    """Query Falcon API for entity linking."""
    api_url = "https://labs.tib.eu/falcon/api?mode=long" if kb == "dbpedia" else "https://labs.tib.eu/falcon/falcon2/api?mode=long"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error: {e}")
        return None


def query_dbpedia(head_ent, tail_ent):
    """Query DBpedia for relationships between entities."""
    sparql = SPARQLWrapper("https://dbpedia.data.dice-research.org/sparql")
    query = f"""
    SELECT ?item 
    WHERE {{
        <{head_ent}> ?item <{tail_ent}>.
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    try:
        return [(result['item']['value']) for result in results["results"]["bindings"]]
    except (TypeError, KeyError) as e:
        print(f"Error accessing JSON data: {e}")
        return None


def query_wikidata(head_ent, tail_ent):
    """Query Wikidata for relationships between entities."""
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "YourAppName/1.0 (your-email@example.com)"}
    query = f"""
    SELECT ?item 
    WHERE {{
        <{head_ent}> ?item <{tail_ent}>.
    }}
    """
    params = {"query": query, "format": "json"}
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        try:
            results = response.json()
            return [(result['item']['value'], result.get('itemLabel', {}).get('value', '')) for result in results["results"]["bindings"]]
        except ValueError:
            print("Response is not valid JSON.")
            return None
    else:
        print(f"Error: Status code {response.status_code}")
        return None


def query_with_retries(api_function, *args):
    """Attempts to call an API function with retry logic."""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = api_function(*args)
            if response is not None:
                return response
        except Exception as e:
            print(f"API call failed with error: {e}")

        retries += 1
        print(f"Retrying in {RETRY_DELAY} seconds... (Attempt {retries}/{MAX_RETRIES})")
        time.sleep(RETRY_DELAY)

    print("Maximum retries exceeded. Stopping further attempts.")
    return None


def calculate_score_for_triple(triple, extracted_sentences):
    """Calculate a score for the given triple."""
    head_entity = triple['Head Entity']
    tail_entity = triple['Tail Entity']
    evidence_sentence_ids = triple['Evidence Sentences']
    relation = triple['Relation']
    score = 0

    if verify_entities_in_evidence_sentences(extracted_sentences, head_entity, tail_entity, evidence_sentence_ids):
        score += 0.5
        
        head_wikidata_response = query_with_retries(query_falcon_api, "wikidata", head_entity)
        tail_wikidata_response = query_with_retries(query_falcon_api, "wikidata", tail_entity)

        head_link = head_wikidata_response['entities_wikidata'][0]['URI'] if head_wikidata_response and 'entities_wikidata' in head_wikidata_response and head_wikidata_response['entities_wikidata'] else None
        tail_link = tail_wikidata_response['entities_wikidata'][0]['URI'] if tail_wikidata_response and 'entities_wikidata' in tail_wikidata_response and tail_wikidata_response['entities_wikidata'] else None

        if head_link and tail_link:
            score += 0.5
            wikidata_relation = query_wikidata(head_link, tail_link)

            if wikidata_relation:
                for rel in wikidata_relation:
                    if rel[0] == 'http://www.wikidata.org/prop/direct/' + relation:
                        score += 1
                        break
        else:
            score -= 0.5

        head_dbpedia_response = query_with_retries(query_falcon_api, "dbpedia", head_entity)
        tail_dbpedia_response = query_with_retries(query_falcon_api, "dbpedia", tail_entity)

        head_dbpedia_uri = head_dbpedia_response['entities'][0]['URI'] if head_dbpedia_response and 'entities' in head_dbpedia_response and head_dbpedia_response['entities'] else None
        tail_dbpedia_uri = tail_dbpedia_response['entities'][0]['URI'] if tail_dbpedia_response and 'entities' in tail_dbpedia_response and tail_dbpedia_response['entities'] else None

        if head_dbpedia_uri and tail_dbpedia_uri:
            score += 0.5
            dbpedia_relation = query_dbpedia(head_dbpedia_uri, tail_dbpedia_uri)

            if dbpedia_relation:
                for rel in dbpedia_relation:
                    if get_wikidata_identifier_from_dbpedia(rel) == relation:
                        score += 1
    else:
        score -= 2

    return score


def main():
    """Main function to process triples and calculate scores."""
    parser = argparse.ArgumentParser(description="Process triples and calculate scores.")
    parser.add_argument("--triples_file", type=str, required=True, help="Path to triples CSV file.")
    parser.add_argument("--sentences_file", type=str, required=True, help="Path to evidence sentences file.")
    args = parser.parse_args()

    triples = pd.read_csv(args.triples_file).to_dict(orient="records")
    extracted_sentences = pd.read_csv(args.sentences_file).to_dict(orient="records")

    for triple in triples:
        score = calculate_score_for_triple(triple, extracted_sentences)
        print(f"Score for triple {triple}: {score}")


if __name__ == "__main__":
    main()

