import requests
import json
import os
from bs4 import BeautifulSoup

# --- Your provided information ---
atlassian_domain = "apiit.atlassian.net"
api_username = "TP083057@mail.apu.edu.my"
api_token = "ATATT3xFfGF0GzrHEcrKMIbTgVyMicdOmh2_grmCPvhcGicr23FPyWZNyaMmOAf8_0eGaYUAyVbwyOSuT-9eHoJsgSnBy5XAVV2tuuZv1Z4rYRf_52-KTaH0idwfg6y-pSACC7gvs4HndodDyc8OylNuXGcBzpAH5-OiiNPe9lB1dAwuulUXPA=72D838BC"

# List of space keys to scrape
space_keys = ["ITSM", "LIB", "LNO", "VISA", "BUR", "AA", ]

# --- API and file configuration ---
confluence_url = f"https://{atlassian_domain}/wiki/rest/api"
base_wiki_url = f"https://{atlassian_domain}/wiki"

def get_all_pages_in_space(space_key):
    """Retrieves all pages from a given Confluence space using the API."""
    pages = []
    start = 0
    limit = 100
    while True:
        url = f"{confluence_url}/space/{space_key}/content?expand=page,version&start={start}&limit={limit}"
        response = requests.get(url, auth=(api_username, api_token))
        response.raise_for_status()
        data = response.json()
        pages.extend(data.get('page', {}).get('results', []))
        
        # Check for more pages
        if len(pages) >= data.get('page', {}).get('size', 0):
            break
        start += limit
    
    return pages

def get_page_content(page_id):
    """Fetches the content of a single page by its ID."""
    url = f"{confluence_url}/content/{page_id}?expand=body.storage"
    response = requests.get(url, auth=(api_username, api_token))
    response.raise_for_status()
    page_data = response.json()
    
    title = page_data.get('title', 'No Title Found')
    # Use the 'body.storage' to get the HTML-like content
    html_content = page_data.get('body', {}).get('storage', {}).get('value', '')
    
    # Use BeautifulSoup to clean the HTML and get plain text
    soup = BeautifulSoup(html_content, 'html.parser')
    clean_text = soup.get_text(separator='\n', strip=True)
    
    return title, clean_text, page_data.get('_links', {}).get('webui')

def main():
    for space_key in space_keys:
        output_dir = f"scraped_articles_{space_key}"
        
        try:
            print(f"Fetching pages from space: {space_key}...")
            all_pages = get_all_pages_in_space(space_key)
            
            if not all_pages:
                print(f"No pages found in space '{space_key}'. Skipping.")
                continue

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # --- Option 1: Save each article as a separate .txt file ---
            for page in all_pages:
                page_id = page['id']
                page_title = page['title']
                print(f"Scraping page ID: {page_id}, Title: {page_title}")
                
                title, content, relative_url = get_page_content(page_id)
                
                # Make the filename safe by replacing invalid characters
                safe_filename = "".join([c for c in title if c.isalnum() or c in (' ', '_')]).rstrip()
                
                with open(os.path.join(output_dir, f"{safe_filename}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Title: {title}\n\n")
                    f.write(f"URL: {base_wiki_url}{relative_url}\n\n")
                    f.write(content)
            
            print(f"\nScraping complete for '{space_key}'. Content saved to individual files in '{output_dir}'.")
            
            # --- Option 2: Save all articles to a single JSON file (for RAG) ---
            json_file = f"scraped_data_{space_key}.json"
            all_data = []
            for page in all_pages:
                page_id = page['id']
                title, content, relative_url = get_page_content(page_id)
                all_data.append({
                    "title": title,
                    "url": f"{base_wiki_url}{relative_url}",
                    "content": content
                })
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, ensure_ascii=False, indent=4)
            
            print(f"Also saved all data to a single JSON file: {json_file}")

        except requests.exceptions.HTTPError as err:
            print(f"HTTP Error for space '{space_key}': {err}")
            print("Please double-check your username, API token, and space key.")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()