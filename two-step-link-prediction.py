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
    """
    Perform fuzzy matching of an entity against detected entities.
    
    Args:
        detected_entities: List of detected entity strings
        entity: Target entity string to match
        threshold: Minimum similarity score (0-100) to consider a match
        
    Returns:
        Boolean indicating if a match was found
    """
    for detected_entity in detected_entities:
        if fuzz.partial_ratio(detected_entity, entity) >= threshold:
            return True
    return False

def verify_entities_in_evidence_sentences(sentences, head_entity, tail_entity, evidence_sentence_ids):
    """
    Verify the presence of entities in evidence sentences.
    
    Args:
        sentences: List of sentence data dictionaries
        head_entity: Head entity string
        tail_entity: Tail entity string
        evidence_sentence_ids: List of sentence IDs that contain evidence
        
    Returns:
        Boolean indicating if both entities are found in evidence sentences
    """
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
    """
    Query Falcon API for entity linking.
    
    Args:
        kb: Knowledge base to query ("dbpedia" or "wikidata")
        text: Entity text to link
        
    Returns:
        JSON response from Falcon API or None if request fails
    """
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
    """
    Query DBpedia for relationships between entities.
    
    Args:
        head_ent: DBpedia URI of head entity
        tail_ent: DBpedia URI of tail entity
        
    Returns:
        List of relation URIs between entities or None if query fails
    """
    sparql = SPARQLWrapper("https://dbpedia.data.dice-research.org/sparql")
    query = f"""
    SELECT ?item 
    WHERE {{
        <{head_ent}> ?item <{tail_ent}>.
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        results = sparql.query().convert()
        return [(result['item']['value']) for result in results["results"]["bindings"]]
    except (TypeError, KeyError) as e:
        print(f"Error accessing JSON data: {e}")
        return None

def query_wikidata(head_ent, tail_ent):
    """
    Query Wikidata for relationships between entities.
    
    Args:
        head_ent: Wikidata URI of head entity
        tail_ent: Wikidata URI of tail entity
        
    Returns:
        List of relation URIs between entities or None if query fails
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "DocRE-RLKGf/1.0 (research@example.com)"}
    query = f"""
    SELECT ?item ?itemLabel
    WHERE {{
        <{head_ent}> ?item <{tail_ent}>.
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    params = {"query": query, "format": "json"}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            results = response.json()
            return [(result['item']['value'], result.get('itemLabel', {}).get('value', '')) 
                   for result in results["results"]["bindings"]]
        else:
            print(f"Error: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error querying Wikidata: {e}")
        return None

def query_with_retries(api_function, *args):
    """
    Attempts to call an API function with retry logic.
    
    Args:
        api_function: Function to call
        *args: Arguments to pass to the function
        
    Returns:
        Response from the API function or None if all retries fail
    """
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

def get_wikidata_identifier_for_relation(relation_uri):
    """
    Extract the Wikidata property identifier from a relation URI.
    
    Args:
        relation_uri: Full Wikidata relation URI
        
    Returns:
        Property identifier (e.g., 'P31') or None if not found
    """
    match = re.search(r'P\d+$', relation_uri)
    if match:
        return match.group(0)
    return None

def calculate_score_for_triple(triple, extracted_sentences, confidence_param=0.5):
    """
    Calculate a score for the given triple based on KG verification.
    
    Args:
        triple: Dictionary containing triple information
        extracted_sentences: List of sentence data dictionaries
        confidence_param: Confidence parameter for scoring (default: 0.5)
        
    Returns:
        Numerical score for the triple
    """
    head_entity = triple['Head Entity']
    tail_entity = triple['Tail Entity']
    evidence_sentence_ids = triple['Evidence Sentences']
    relation = triple['Relation']
    score = 0

    # Step 1: Verify entities in evidence sentences
    if verify_entities_in_evidence_sentences(extracted_sentences, head_entity, tail_entity, evidence_sentence_ids):
        score += 0
        
        # Step 2: Entity Linking with Wikidata
        head_wikidata_response = query_with_retries(query_falcon_api, "wikidata", head_entity)
        tail_wikidata_response = query_with_retries(query_falcon_api, "wikidata", tail_entity)

        head_link = None
        tail_link = None
        
        if head_wikidata_response and 'entities_wikidata' in head_wikidata_response and head_wikidata_response['entities_wikidata']:
            head_link = head_wikidata_response['entities_wikidata'][0]['URI']
            
        if tail_wikidata_response and 'entities_wikidata' in tail_wikidata_response and tail_wikidata_response['entities_wikidata']:
            tail_link = tail_wikidata_response['entities_wikidata'][0]['URI']

        if head_link and tail_link:
            score += confidence_param
            
            # Step 3: Query relation in Wikidata
            wikidata_relations = query_with_retries(query_wikidata, head_link, tail_link)

            if wikidata_relations:
                relation_found = False
                for rel in wikidata_relations:
                    rel_id = get_wikidata_identifier_for_relation(rel[0])
                    if rel_id and rel_id == relation:
                        score += 2 * confidence_param
                        relation_found = True
                        break
                
                if not relation_found:
                    score -= confidence_param
            else:
                # No relations found between entities
                score -= confidence_param
        else:
            # One or both entities not found in Wikidata
            score -= 4 * confidence_param
            
        # Step 4: Try DBpedia as fallback
        if score <= 0:
            head_dbpedia_response = query_with_retries(query_falcon_api, "dbpedia", head_entity)
            tail_dbpedia_response = query_with_retries(query_falcon_api, "dbpedia", tail_entity)

            head_dbpedia_uri = None
            tail_dbpedia_uri = None
            
            if head_dbpedia_response and 'entities' in head_dbpedia_response and head_dbpedia_response['entities']:
                head_dbpedia_uri = head_dbpedia_response['entities'][0]['URI']
                
            if tail_dbpedia_response and 'entities' in tail_dbpedia_response and tail_dbpedia_response['entities']:
                tail_dbpedia_uri = tail_dbpedia_response['entities'][0]['URI']

            if head_dbpedia_uri and tail_dbpedia_uri:
                score += confidence_param
                dbpedia_relations = query_with_retries(query_dbpedia, head_dbpedia_uri, tail_dbpedia_uri)

                if dbpedia_relations:
                    # For simplicity, we assume a match if any relation is found
                    # In a real implementation, you would map DBpedia relations to your schema
                    score += confidence_param
    else:
        # Entities not found in evidence sentences
        score -= 2 * confidence_param

    return score

def calculate_prediction_score(prediction, extracted_sentences, confidence_param=0.5):
    """
    Calculate an aggregate score for all triples in a prediction.
    
    Args:
        prediction: List of triple dictionaries
        extracted_sentences: List of sentence data dictionaries
        confidence_param: Confidence parameter for scoring
        
    Returns:
        Aggregate score for the prediction
    """
    total_score = 0
    for triple in prediction:
        triple_score = calculate_score_for_triple(triple, extracted_sentences, confidence_param)
        total_score += triple_score
    
    return total_score

def process_predictions_file(predictions_file, sentences_file, output_file, confidence_param=0.5):
    """
    Process a file of predictions and calculate scores.
    
    Args:
        predictions_file: Path to CSV file with predictions
        sentences_file: Path to CSV file with sentences
        output_file: Path to save scored predictions
        confidence_param: Confidence parameter for scoring
        
    Returns:
        DataFrame with predictions and their scores
    """
    predictions_df = pd.read_csv(predictions_file)
    sentences_df = pd.read_csv(sentences_file)
    
    # Group predictions by document ID
    grouped_predictions = predictions_df.groupby('Document ID')
    
    results = []
    for doc_id, group in grouped_predictions:
        doc_sentences = sentences_df[sentences_df['Document ID'] == doc_id].to_dict('records')
        
        # Get all predictions for this document
        doc_predictions = []
        for _, row in group.iterrows():
            triple = {
                'Head Entity': row['Head Entity'],
                'Tail Entity': row['Tail Entity'],
                'Relation': row['Relation'],
                'Evidence Sentences': eval(row['Evidence Sentences']) if isinstance(row['Evidence Sentences'], str) else row['Evidence Sentences']
            }
            doc_predictions.append(triple)
        
        # Calculate score for this document's predictions
        score = calculate_prediction_score(doc_predictions, doc_sentences, confidence_param)
        
        results.append({
            'Document ID': doc_id,
            'Predictions': doc_predictions,
            'Score': score
        })
    
    results_df = pd.DataFrame(results)
    
    if output_file:
        results_df.to_csv(output_file, index=False)
    
    return results_df

def main():
    """Main function to process triples and calculate scores."""
    parser = argparse.ArgumentParser(description="Process triples and calculate scores based on knowledge graph feedback.")
    parser.add_argument("--predictions_file", type=str, required=True, help="Path to predictions CSV file.")
    parser.add_argument("--sentences_file", type=str, required=True, help="Path to evidence sentences file.")
    parser.add_argument("--output_file", type=str, default="scored_predictions.csv", help="Path to save scored predictions.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence parameter for scoring.")
    args = parser.parse_args()

    results = process_predictions_file(
        args.predictions_file, 
        args.sentences_file, 
        args.output_file,
        args.confidence
    )
    
    print(f"Processed {len(results)} documents. Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
