# # Load model directly
# from transformers import AutoTokenizer, AutoModelForTokenClassification
# import pandas as pd
# import spacy
#
# nlp = spacy.load("en_core_web_trf")
#
# from transformers import pipeline
# from nltk import sent_tokenize
# import nltk
# import re
# import itertools
# import networkx as nx
# # nltk.download('punkt_tab')
# from pyvis.network import Network
# import pyvis
# import webbrowser
#
# # pipe = pipeline("token-classification", model="dslim/bert-base-NER")
#
# # tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
# # model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
#
#
# tokenizer = AutoTokenizer.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
# model = AutoModelForTokenClassification.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
#
# df = pd.read_csv("../naruto_scraper/data/Processed_files/Processed_subtitles.csv")
#
# # text = df['Subtitles'][50]
# # print(text)
#
#
# ### Custome method to get the person names from the NER Model
#
#
# def normalize(text):
#     # Remove non-alphabetic characters and convert to lowercase
#     return re.sub(r'[^a-zA-Z]', '', text).lower()
#
#
# def remove_similar_characters(char_list):
#     seen = set()  # To track normalized versions of strings
#     unique_list = []
#
#     for char in char_list:
#         normalized_char = normalize(char)
#
#         # If the normalized version is not seen, add the original to the unique list
#         if normalized_char not in seen:
#             unique_list.append(char)
#             seen.add(normalized_char)
#
#     return unique_list
#
#
# def create_combinations(char_list):
#     # Generate all combinations of 2 characters from the list
#     combinations = list(itertools.combinations(char_list, 2))
#     # Convert tuples to lists for a 2D list format
#     return [sorted(list(combo)) for combo in combinations]
#
#
# def create_ner(script):
#     script_sent = sent_tokenize(script)
#
#     # Loop to merge sentences 3 at a time
#     merged_sentences = []
#     for i in range(0, len(script_sent), 10):
#         # Merge 3 sentences at a time using slicing
#         if i + 10 <= len(script_sent):
#             merged = ' '.join(script_sent[i:i + 10])
#             merged_sentences.append(merged)
#         else:
#             merged = ' '.join(script_sent[i::])
#             merged_sentences.append(merged)
#
#     ner_output = []
#     ner_output_processed = [[]]
#
#     for sentence in merged_sentences:
#         doc = nlp(sentence)
#         ners = set()
#         for ent in doc.ents:
#             if ent.label_ == 'PERSON':
#                 name = ent.text
#                 ners.add(name)
#         ner_output.append(list(ners))
#     ner_output = [ele for ele in ner_output if len(ele) >= 2]
#
#     for ele in ner_output:
#         processed_char_list = remove_similar_characters(ele)
#         if len(processed_char_list) >= 2:
#             processed_char_list = create_combinations(processed_char_list)
#         else:
#             processed_char_list = [[]]
#
#         ner_output_processed = ner_output_processed + processed_char_list
#     ner_output_processed = [ele for ele in ner_output_processed if len(ele) >= 2]
#
#     print(ner_output)
#     print(ner_output_processed)
#
#     return ner_output_processed
#
#
# def generate_char_network(episode=None):
#     if episode is not None:
#         text = df['Subtitles'][episode]
#         character_data = create_ner(text)
#     else:
#         text = df['Subtitles'][20]
#         character_data = create_ner(text)
#
#     character_df = pd.DataFrame(character_data, columns=['Character 1', 'Character 2'])
#
#     character_df = character_df.groupby(['Character 1', 'Character 2']).size().reset_index(name='Count')
#     print(character_df)
#
#     ## create character network
#     G = nx.from_pandas_edgelist(
#         character_df,
#         source='Character 1',
#         target='Character 2',
#         edge_attr='Count',
#         create_using=nx.Graph()
#     )
#
#     ### Create the Visualization
#
#     net = Network(notebook=False)
#     node_degree = dict(G.degree)
#     nx.set_node_attributes(G, node_degree, 'size')
#     net.from_nx(G)
#
#     html = net.generate_html()
#     html = html.replace("'", "\"")
#
#     output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
#         display-capture; encrypted-media;" sandbox="allow-modals allow-forms
#         allow-scripts allow-same-origin allow-popups
#         allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
#         allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
#
#     with open("sample_output.html", "w") as html_file:
#         html_file.write(output_html)
#
#     print("HTML File has been updated")
#
#
#
# # import webbrowser
# # webbrowser.open('sample_output.html')
#
# # net.show("Naruto_char_network.html")
#
#
# # doc = nlp(text)
# #
# # for ent in doc.ents:
# #     print(ent.text, ent.label_)
#
# # nlp = pipeline("ner", model=model, tokenizer=tokenizer)
# # example = "My name is Wolfgang and I live in Berlin"
# #
# # ner_results = nlp(text)
# # print(ner_results)


from transformers import AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
import spacy
from nltk import sent_tokenize
import nltk
import re
import itertools
import networkx as nx
from pyvis.network import Network
from glob import glob
files = glob('../naruto_scraper/data/Processed_files/Processed_subtitles.csv"')


class CharacterNetworkGenerator:
    def __init__(self, model_name="eventdata-utd/conflibert-named-entity-recognition",
                 csv_path="../naruto_scraper/data/Processed_files/Processed_subtitles.csv"):
        self.nlp = spacy.load("en_core_web_trf")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.df = pd.read_csv(files[0])

    def normalize(self, text):
        return re.sub(r'[^a-zA-Z]', '', text).lower()

    def remove_similar_characters(self, char_list):
        seen = set()
        unique_list = []
        for char in char_list:
            normalized_char = self.normalize(char)
            if normalized_char not in seen:
                unique_list.append(char)
                seen.add(normalized_char)
        return unique_list

    def create_combinations(self, char_list):
        combinations = list(itertools.combinations(char_list, 2))
        return [sorted(list(combo)) for combo in combinations]

    def create_ner(self, script):
        script_sent = sent_tokenize(script)
        merged_sentences = [' '.join(script_sent[i:i + 10]) for i in range(0, len(script_sent), 10)]
        ner_output = []

        for sentence in merged_sentences:
            doc = self.nlp(sentence)
            ners = {ent.text for ent in doc.ents if ent.label_ == 'PERSON'}
            ner_output.append(list(ners))

        ner_output = [ele for ele in ner_output if len(ele) >= 2]
        ner_output_processed = [[]]

        for ele in ner_output:
            processed_char_list = self.remove_similar_characters(ele)
            if len(processed_char_list) >= 2:
                processed_char_list = self.create_combinations(processed_char_list)
            else:
                processed_char_list = [[]]
            ner_output_processed += processed_char_list

        ner_output_processed = [ele for ele in ner_output_processed if len(ele) >= 2]
        return ner_output_processed

    def generate_char_network(self, episode=20):
        # episode = int(episode)

        text = self.df['Subtitles'][episode]
        character_data = self.create_ner(text)
        character_df = pd.DataFrame(character_data, columns=['Character 1', 'Character 2'])
        character_df = character_df.groupby(['Character 1', 'Character 2']).size().reset_index(name='Count')

        G = nx.from_pandas_edgelist(
            character_df,
            source='Character 1',
            target='Character 2',
            edge_attr='Count',
            create_using=nx.Graph()
        )

        net = Network(notebook=False)
        node_degree = dict(G.degree)
        nx.set_node_attributes(G, node_degree, 'size')
        net.from_nx(G)

        html = net.generate_html().replace("'", "\"")
        output_html = f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera;
            display-capture; encrypted-media;" sandbox="allow-modals allow-forms
            allow-scripts allow-same-origin allow-popups
            allow-top-navigation-by-user-activation allow-downloads" allowfullscreen=""
            allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

        with open("sample_output.html", "w") as html_file:
            html_file.write(output_html)

        print("HTML File has been updated")
