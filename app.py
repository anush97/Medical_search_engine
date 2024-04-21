from flask import Flask, request, jsonify, render_template
import requests
import xml.etree.ElementTree as ET
from dotenv import load_dotenv
import os
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
import openai

app = Flask(__name__)
load_dotenv()

# Retrieve API key from environment variable
NCBI_API_KEY = os.getenv('NCBI_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = openai_api_key


# Load local LLM model
local_llm = "BioMistral-7B.Q4_K_M.gguf"
llm = LlamaCpp(
    model_path=local_llm,
    temperature=0.4,
    max_tokens=500,  
    top_p=1
)
print("LLM Initialized....")

# Define the prompt templates for keyword generation and summarization
keyword_prompt_template = """
Your task is to pick or generate three most relevant keywords to search for a relevant medical research papers on PubMed based on given query : {query}
"""

final_summary_prompt_template = """Answer the query : {query} asked by the user truthfully based on the given articles. If the documents don't contain an answer, use your existing knowledge base.
{combined_summaries}
"""

summary_prompt_template = """Summarize the following medical article's abstract in two lines answering user query : {query} or highlighting relevant text according to user query : {query} truthfully based on the given articles.
{abstracts}
"""

class PubMedSearchPipeline:
    def __init__(self, llm):
        """Initialize the search pipeline with an LLM instance."""
        self.llm = llm
        
    def generate_keywords(self, query):
        """Generate keywords from a user query using the LLM."""
        keyword_prompt = keyword_prompt_template.format(query=query)
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt= keyword_prompt ,
                temperature=0.4,
                max_tokens=100
        )
            response_text = response.choices[0].text.strip()
            # Extract keywords
            keywords = []
            for line in response_text.split('\n'):
                if line.strip():
                    # Remove digits and dots at the start of each line
                    cleaned_keyword = line.strip().split('. ', 1)[-1]
                    keywords.append(cleaned_keyword)
            #print("Extracted Keywords:", keywords)
            return keywords[:3]  # Return the first three cleaned keywords
        except Exception as e:
            print(f"Error generating keywords using GPT-3.5 Turbo: {e}")
            return []

    def search_pubmed(self, keywords):
        """ Search PubMed database for articles matching the given keywords. """
        pubmed_ids = []
        for keyword in keywords:
            #print("Original keyword:", keyword)
            # Strip spaces and replace internal spaces with '+'
            formatted_keyword = keyword.strip().replace(' ', '+')
            #print("Formatted keyword:", formatted_keyword)
            search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={formatted_keyword}&retmode=json&retmax=5&api_key={NCBI_API_KEY}"
            try:
                response = requests.get(search_url).json()
                ids = response['esearchresult']['idlist']
                pubmed_ids.extend(ids)
                #print("pubmed_ids - ",len(pubmed_ids))
                if len(pubmed_ids) >=15:
                    break  
            except Exception as e:
                print(f"Error fetching PubMed IDs for keyword '{formatted_keyword}': {e}")
        return list(set(pubmed_ids))[:15]
            
    def generate_final_summary(self, combined_summaries, query):
        """Generate a final summary from combined summaries."""
        if not combined_summaries:
            return "No relevant information found based on the query."

        prompt = final_summary_prompt_template.format(combined_summaries=combined_summaries, query=query)
        try:
            final_response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=150,  
                temperature=0.4  
            )
            return final_response.choices[0].text.strip() if final_response.choices[0].text.strip() else "No conclusive summary could be generated."

            '''final_response = self.llm.invoke(prompt)
            return final_response.strip() if final_response.strip() else "No conclusive summary could be generated."'''
        except Exception as e:
            print(f"Error generating final summary: {e}")
            return "Error in generating final response."

    def fetch_and_process_abstracts(self, pubmed_ids, query):
        """Fetch abstracts for the given PubMed IDs and process them using LLM."""
        combined_summaries = []
        for pubmed_id in pubmed_ids:
            abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml&api_key={NCBI_API_KEY}"
            try:
                response = requests.get(abstract_url)
                xml_data = response.text
                root = ET.fromstring(xml_data)
                abstract_text = " ".join([elem.text for elem in root.findall('.//Abstract/AbstractText') if elem.text])
                prompt = summary_prompt_template.format(abstracts=abstract_text , query = query)
                '''summary_response = self.llm.invoke(prompt)
                if summary_response.strip():
                    combined_summaries.append(summary_response.strip())'''
                summary_response = openai.Completion.create(
                    model="gpt-3.5-turbo-instruct",
                    prompt=prompt,
                    max_tokens=3000,  
                    temperature=0.3 
                )
                if summary_response.choices[0].text.strip():
                    #print("summary_response.choices[0].text.strip()", summary_response.choices[0].text.strip())
                    combined_summaries.append(summary_response.choices[0].text.strip())
            except Exception as e:
                print(f"Error fetching or processing abstract for PubMed ID {pubmed_id}: {e}")
        return " ".join(combined_summaries)


    def fetch_abstracts(self, pubmed_ids):
        abstracts = []
        for pubmed_id in pubmed_ids:
            abstract_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml&api_key={NCBI_API_KEY}"
            try:
                response = requests.get(abstract_url)
                xml_data = response.text
                root = ET.fromstring(xml_data)
                abstract_text = ""
                for abstract in root.findall('.//Abstract/AbstractText'):
                    if abstract.text:
                        abstract_text += abstract.text + " "
                if abstract_text:
                    abstracts.append(abstract_text.strip())
            except Exception as e:
                print(f"Error fetching abstract for PubMed ID {pubmed_id}: {e}")
        return " ".join(abstracts) 
    

    '''def summarize(self, abstracts):
        if not abstracts:
            return "No valid data to summarize."

        # Simple tokenization assuming space as delimiter; you may need a more sophisticated tokenizer
        tokens = abstracts.split()
        max_tokens_per_request = 500  # Adjusted to be safely below the model's token limit
        chunks = []

        current_chunk = []
        current_length = 0

        for token in tokens:
            if current_length + len(token.split()) > max_tokens_per_request:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            current_chunk.append(token)
            current_length += len(token.split())

        if current_chunk:
            chunks.append(' '.join(current_chunk)) 

        summaries = []
        for chunk in chunks:
            prompt = summary_prompt_template.format(abstracts=chunk)
            try:
                summary_response = self.llm.invoke(prompt)
                if summary_response.strip():
                    summaries.append(summary_response.strip())
            except Exception as e:
                print(f"Error summarizing chunk: {e}")
                summaries.append("Error in summarizing part of the text.")

        return " ".join(summaries) if summaries else "No valid summary generated."'''


@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('chat.html')

@app.route('/search', methods=['POST'])
def search_papers():
    """Handle the search query from the user."""
    query = request.form.get('query')
    if not query:
        return jsonify({'error': 'Query is required'}), 400

    pipeline = PubMedSearchPipeline(llm)
    keywords = pipeline.generate_keywords(query)
    if not keywords:
        return jsonify({'error': 'No keywords generated from the query.'}), 400

    pubmed_ids = pipeline.search_pubmed(keywords)
    if not pubmed_ids:
        return jsonify({'error': 'No articles found for the given keywords.'}), 404

    combined_summaries = pipeline.fetch_and_process_abstracts(pubmed_ids, query)
    final_summary = pipeline.generate_final_summary(combined_summaries, query)
    article_links = {pid: f"https://pubmed.ncbi.nlm.nih.gov/{pid}/" for pid in pubmed_ids[:3]}
    return jsonify({'summary': final_summary, 'top_articles': article_links})


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Page not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
