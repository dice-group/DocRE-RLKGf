def generate_prompt_with_answers(data):
    """
    Generates a prompt with responses from a data dictionary for fine-tuning, including answers from labels.

    Parameters:
        data (dict): The data dictionary with keys 'title', 'sents', 'vertexSet', and 'labels'.

    Returns:
        str: A formatted prompt with responses.
    """
    title = data['title']
    
    # Formatting sentences
    sentences = "\n".join([f"Sentence {i}: {' '.join(sent)}" for i, sent in enumerate(data['sents'])])
    
    # Formatting entities
    entities = []
    for i, entity in enumerate(data['vertexSet']):
        mentions = []
        for mention in entity:
            mention_text = (
                f"'{mention['name']}' in sentence {mention['sent_id']} "
                f"(position {mention['pos'][0]}-{mention['pos'][1]}), type: {mention['type']}"
            )
            mentions.append(mention_text)
        entities.append(f"Entity {i + 1}: Mentions: {', '.join(mentions)}")
    
    formatted_entities = "\n".join(entities)

    # Building the initial prompt
    prompt = f"""
    Given a document titled **"{title}"**, analyze the following sentences to identify relationships between entities.
    Specify the head entity, tail entity, the relationship, and any evidence sentences supporting this relationship.

    ### Document
    **Sentences:**  
    {sentences}

    **Entities:**  
    {formatted_entities}

    ### Task
    Identify the relationships as follows:
    - **Head Entity**: Main entity in the relationship.
    - **Tail Entity**: Secondary entity in the relationship.
    - **Relation**: Type of relationship between entities.
    - **Evidence Sentence(s)**: Sentence ID(s) containing evidence for the relationship.

    ### Example Output
    """

    # Adding example responses based on labels
    example_responses = []
    for label in data['labels']:
        head_entity_idx = label['h']
        tail_entity_idx = label['t']
        relation = label['r']
        evidence_sents = label['evidence']
        
        head_entity_name = data['vertexSet'][head_entity_idx][0]['name']
        tail_entity_name = data['vertexSet'][tail_entity_idx][0]['name']
        evidence_sentences = ", ".join(map(str, evidence_sents))

        example_response = f"""
        - **Head Entity**: {head_entity_name}
        - **Tail Entity**: {tail_entity_name}
        - **Relation**: {relation}
        - **Evidence Sentence(s)**: {evidence_sentences}
        """
        example_responses.append(example_response.strip())
    
    # Join all responses into the prompt
    prompt += "\n\n".join(example_responses)
    
    return prompt.strip()

# Example usage with the given data
data = {"title": "DW_1213203", "vertexSet": [[{"name": "East Germany", "pos": [143, 145], "sent_id": 0, "type": "type::location"}, {"name": "GDR", "pos": [213, 214], "sent_id": 0, "type": "type::location"}], [{"name": "EU", "pos": [3, 4], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [12, 13], "sent_id": 0, "type": "type::organization"}, {"name": "European Union", "pos": [88, 90], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [116, 117], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [241, 242], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [335, 336], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [367, 368], "sent_id": 0, "type": "type::organization"}, {"name": "EU", "pos": [550, 551], "sent_id": 0, "type": "type::organization"}], [{"name": "Mario Monti", "pos": [70, 72], "sent_id": 0, "type": "type::person"}, {"name": "Monti", "pos": [287, 288], "sent_id": 0, "type": "type::person"}], [{"name": "Matthias Platzeck", "pos": [159, 161], "sent_id": 0, "type": "type::person"}, {"name": "Platzeck", "pos": [199, 200], "sent_id": 0, "type": "type::person"}, {"name": "Platzeck", "pos": [426, 427], "sent_id": 0, "type": "type::person"}, {"name": "Platzeck", "pos": [496, 497], "sent_id": 0, "type": "type::person"}], [{"name": "Germany", "pos": [1, 2], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [21, 22], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [49, 50], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [92, 93], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [134, 135], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [172, 173], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [357, 358], "sent_id": 0, "type": "type::location"}, {"name": "Germany", "pos": [391, 392], "sent_id": 0, "type": "type::location"}], [{"name": "Brandenburg", "pos": [68, 69], "sent_id": 0, "type": "type::location"}, {"name": "Brandenburg", "pos": [156, 157], "sent_id": 0, "type": "type::location"}, {"name": "Brandenburg", "pos": [519, 520], "sent_id": 0, "type": "type::location"}, {"name": "Brandenburg", "pos": [525, 526], "sent_id": 0, "type": "type::location"}], [{"name": "European", "pos": [74, 75], "sent_id": 0, "type": "type::other"}], [{"name": "Monday", "pos": [154, 155], "sent_id": 0, "type": "type::time"}], [{"name": "premier", "pos": [158, 159], "sent_id": 0, "type": "type::role"}], [{"name": "Eastern Germany", "pos": [202, 204], "sent_id": 0, "type": "type::location"}, {"name": "Eastern Germany", "pos": [321, 323], "sent_id": 0, "type": "type::location"}], [{"name": "German", "pos": [245, 246], "sent_id": 0, "type": "type::other"}, {"name": "German", "pos": [301, 302], "sent_id": 0, "type": "type::other"}, {"name": "German", "pos": [488, 489], "sent_id": 0, "type": "type::other"}], [{"name": "European Commission", "pos": [254, 256], "sent_id": 0, "type": "type::organization"}, {"name": "European Commission", "pos": [432, 434], "sent_id": 0, "type": "type::organization"}], [{"name": "Jan.1, 2007", "pos": [279, 282], "sent_id": 0, "type": "type::time"}], [{"name": "\u20ac1.25 trillion", "pos": [404, 407], "sent_id": 0, "type": "type::money"}], [{"name": "$1.5", "pos": [408, 410], "sent_id": 0, "type": "type::money"}], [{"name": "Saxony-Anhalt", "pos": [451, 454], "sent_id": 0, "type": "type::location"}], [{"name": "Wolfgang B\u00f6hmer", "pos": [455, 457], "sent_id": 0, "type": "type::person"}], [{"name": "Brussels", "pos": [464, 465], "sent_id": 0, "type": "type::location"}]], "labels": [{"r": "member_of", "h": 2, "t": 1, "evidence": [0]}, {"r": "agent_of", "h": 3, "t": 5, "evidence": [0]}, {"r": "citizen_of", "h": 3, "t": 4, "evidence": [0]}, {"r": "citizen_of-x", "h": 3, "t": 10, "evidence": [0]}, {"r": "head_of_gov", "h": 3, "t": 5, "evidence": [0]}, {"r": "in0", "h": 5, "t": 4, "evidence": [0]}, {"r": "in0-x", "h": 5, "t": 10, "evidence": [0]}, {"r": "gpe0", "h": 10, "t": 4, "evidence": [0]}, {"r": "institution_of", "h": 11, "t": 1, "evidence": [0]}, {"r": "part_of", "h": 11, "t": 1, "evidence": [0]}, {"r": "in0", "h": 15, "t": 4, "evidence": [0]}, {"r": "in0-x", "h": 15, "t": 10, "evidence": [0]}, {"r": "agent_of", "h": 16, "t": 15, "evidence": [0]}, {"r": "citizen_of", "h": 16, "t": 4, "evidence": [0]}, {"r": "citizen_of-x", "h": 16, "t": 10, "evidence": [0]}, {"r": "head_of_gov", "h": 16, "t": 15, "evidence": [0]}], "sents": [["Eastern", "Germany", "'s", "EU", "Funds", "in", "Danger", "of", "Drying", "Up", "\n", "The", "EU", "'s", "competition", "commissioner", "met", "with", "the", "heads", "of", "Germany", "'s", "eastern", "states", "to", "discuss", "future", "financial", "support", "for", "the", "region", ",", "which", "may", "no", "longer", "be", "a", "top", "priority", "following", "enlargement", ".", "\n", "The", "heads", "of", "Germany", "'s", "economically", "depressed", "eastern", "states", "could", "be", "forgiven", "for", "going", "on", "the", "defensive", "prior", "to", "their", "meeting", "in", "Brandenburg", "with", "Mario", "Monti", ",", "the", "European", "commissioner", "for", "competition", ".", "With", "10", "mainly", "former", "communist", "countries", "now", "in", "the", "European", "Union", ",", "eastern", "Germany", "'s", "economic", "problems", "suddenly", "do", "n't", "seem", "so", "bad", ".", "And", "that", "means", "the", "region", "is", "in", "danger", "of", "losing", "some", "of", "the", "EU", "'s", "development", "aid", "that", ",", "together", "with", "the", "so", "-", "called", "\"", "solidarity", "payments", "\"", "from", "western", "Germany", ",", "have", "funded", "reconstruction", "projects", "in", "the", "former", "East", "Germany", ".", "Reconstruction", "far", "from", "over", "During", "the", "meeting", "on", "Monday", ",", "Brandenburg", "'s", "premier", "Matthias", "Platzeck", "said", "it", "would", "be", "difficult", "to", "explain", "to", "people", "in", "eastern", "Germany", "that", "their", "region", "would", "have", "to", "take", "a", "funding", "cutback", ".", "All", "the", "progress", "that", "'s", "been", "made", "in", "recent", "years", "is", "still", "very", "fragile", ",", "Platzeck", "said", ".", "Eastern", "Germany", "had", "been", "completely", "deindustrialized", "after", "the", "collapse", "of", "the", "GDR", ",", "and", "rebuilding", "the", "region", "is", "far", "from", "over", ",", "he", "added", ".", "There", "have", "been", "no", "concrete", "signs", "yet", "as", "to", "how", "big", "a", "cut", "in", "EU", "subsidies", "the", "eastern", "German", "states", "would", "have", "to", "reckon", "with", ".", "The", "European", "Commission", "is", "conducting", "a", "comprehensive", "review", "of", "its", "regional", "aid", "policy", "in", "order", "to", "introduce", "new", "guidelines", "and", "criteria", "to", "come", "into", "effect", "on", "Jan.1", ",", "2007", ".", "No", "reason", "to", "worry", "Monti", "(", "photo", ")", "said", "it", "was", "important", "for", "him", "to", "talk", "with", "eastern", "German", "leaders", "about", "their", "worries", ".", "He", "assured", "them", "that", "they", "can", "be", "confident", "about", "their", "region", "'s", "future", ".", "Eastern", "Germany", "has", "gone", "from", "the", "outer", "edges", "to", "the", "very", "center", "of", "the", "EU", "--", "a", "development", "that", "would", "soon", "manifest", "itself", "in", "further", "growth", "and", "development", ",", "he", "said", ".", "Statistically", "speaking", ",", "eastern", "Germany", "is", "no", "longer", "among", "the", "poorest", "regions", "in", "the", "EU", ",", "now", "that", "the", "bloc", "has", "expanded", "eastwards", "to", "include", "several", "ex", "-", "communist", "states", "with", "struggling", "economies", ".", "However", ",", "unemployment", "in", "Germany", "'s", "eastern", "states", "is", "over", "20", "percent", ",", "and", "despite", "an", "estimated", "\u20ac", "1.25", "trillion", "(", "$", "1.5", ")", "in", "aid", ",", "the", "economic", "gap", "separating", "east", "and", "west", "has", "n't", "been", "closed", ".", "Platzeck", "said", "that", "parts", "of", "the", "European", "Commission", "'s", "plan", "to", "reform", "its", "regional", "aid", "policy", "will", "only", "exacerbate", "the", "problem", ".", "The", "premier", "of", "Saxony", "-", "Anhalt", ",", "Wolfgang", "B\u00f6hmer", ",", "said", "that", "without", "financial", "help", "from", "Brussels", "his", "state", "would", "face", "a", "budget", "emergency", ".", "Last", "month", ",", "a", "government", "-", "commissioned", "report", "came", "to", "the", "conclusion", "that", "the", "east", "German", "reconstruction", "project", "was", "a", "failure", ".", "But", "Platzeck", "said", "accusations", "of", "ineffectiveness", "are", "unfair", ".", "He", "said", "47", "of", "the", "50", "large", "-", "scale", "development", "projects", "in", "his", "state", "of", "Brandenburg", "were", "working", "very", "well", ".", "Brandenburg", "'s", "state", "government", "has", "already", "divided", "the", "region", "into", "two", "zones", ",", "so", "that", "it", "can", "at", "least", "apply", "for", "the", "highest", "level", "of", "EU", "funding", "for", "the", "poorer", "northern", "half", "."]]}

# Generate and print the prompt with responses
print(generate_prompt_with_answers(data))

