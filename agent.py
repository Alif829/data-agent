import openai
import pandas as pd
from playwright.sync_api import sync_playwright
from serpapi import GoogleSearch
from PyPDF2 import PdfReader
from io import BytesIO
import re
from typing import List
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from transformers import pipeline

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize summarization pipeline
summarizer = pipeline('summarization')

def search_relevant_pages(city: str) -> List[str]:
    query = f"{city} homicide statistics site:.gov OR site:.edu OR site:.org"
    params = {
        "q": query,
        "api_key": os.getenv('SERPAPI_API_KEY'),
        "engine": "google"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [result['link'] for result in results.get('organic_results', [])]

def scrape_page(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.path.endswith('.pdf'):
        try:
            response = requests.get(url)
            response.raise_for_status()
            with BytesIO(response.content) as open_pdf_file:
                reader = PdfReader(open_pdf_file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() if page.extract_text() else ""
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {url}: {e}")
            return ""
    else:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=60000)
                content = page.content()
                browser.close()

                # Use BeautifulSoup to remove HTML tags
                soup = BeautifulSoup(content, "html.parser")
                text = soup.get_text()
                return text
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return ""

def preprocess_content(content: str) -> str:
    """
    Preprocess content to remove irrelevant text and extract important sections.

    Parameters:
    content (str): The raw scraped content.

    Returns:
    str: The preprocessed, cleaned content.
    """
    # Remove irrelevant sections such as headers, footers, legal disclaimers, etc.
    # Using a simple keyword-based filter to eliminate common irrelevant text
    irrelevant_keywords = ["privacy policy", "terms of service", "subscribe", "cookie policy", "advertisement"]
    lines = content.split('\n')
    relevant_lines = [
        line for line in lines 
        if all(keyword.lower() not in line.lower() for keyword in irrelevant_keywords)
    ]

    # Join filtered lines and apply additional regex for year-based filtering
    cleaned_content = "\n".join(relevant_lines)

    # Use regex to extract sentences containing years (targeting the last 5 years)
    year_pattern = re.compile(r"\b(20[1-2][0-9])\b")  # Matches years 2010-2029
    extracted_sentences = []
    for sentence in cleaned_content.split('.'):
        if year_pattern.search(sentence):
            extracted_sentences.append(sentence.strip())

    return ". ".join(extracted_sentences)

def summarize_content(content: str, max_length: int = 150) -> str:
    """
    Summarize the content to reduce the length before sending it to the LLM.

    Parameters:
    content (str): The content to be summarized.
    max_length (int): The maximum length of the summary.

    Returns:
    str: The summarized content.
    """
    try:
        summary = summarizer(content, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return content

def process_with_llm(content: str, city: str, num_years: int = 5) -> List[dict]:
    # Preprocess the content to reduce token count
    preprocessed_content = preprocess_content(content)

    # Summarize the content to further reduce token usage
    summarized_content = summarize_content(preprocessed_content)

    # Split the summarized content into manageable chunks
    chunks = split_content(summarized_content)
    all_data = []

    for chunk in chunks:
        prompt = (
            f"Below is some information about homicide statistics for the city of {city}. "
            f"Please extract the homicide counts for each year for the past {num_years} years. "
            f"Provide the data in the format: Year, Homicide Count.\n\n"
            f"Content:\n{chunk}"
        )

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        response_text = response['choices'][0]['message']['content'].strip()
        lines = response_text.split('\n')
        for line in lines:
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    year = int(parts[0].strip())
                    count = int(parts[1].strip())
                    all_data.append({"Year": year, "Homicide Count": count})
                except ValueError:
                    print(f"Skipping line due to ValueError: {line}")
                    continue

    return all_data

def split_content(content: str, max_chunk_size: int = 4000) -> List[str]:
    words = content.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def collect_homicide_data(cities: List[str], num_years: int = 5) -> pd.DataFrame:
    all_data = []
    for city in cities:
        urls = search_relevant_pages(city)
        city_data = []
        for url in urls:
            page_content = scrape_page(url)
            if page_content:
                structured_data = process_with_llm(page_content, city, num_years)
                if structured_data:
                    city_data.extend(structured_data)
                    break  # Stop after finding the first valid page

        if city_data:
            for record in city_data:
                all_data.append({"City": city, **record})
    
    return pd.DataFrame(all_data)

def main():
    cities = ["New York", "New Orleans", "Los Angeles"]
    num_years = 5
    homicide_data = collect_homicide_data(cities, num_years)
    if not homicide_data.empty:
        print(homicide_data)
        homicide_data.to_csv("homicide_statistics.csv", index=False)
    else:
        print("No data available to save.")

if __name__ == "__main__":
    main()
