import requests
import os
from bs4 import BeautifulSoup

# --- Information ---
atlassian_domain = "apiit.atlassian.net"
api_username = "TPexample@mail.apu.edu.my"
api_token = "example"

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
    
    # Use BeautifulSoup to clean the HTML and preserve important links
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Convert links to readable format: "text (URL)"
    for link in soup.find_all('a'):
        href = link.get('href', '')
        text = link.get_text(strip=True)
        
        if href and text:
            # Handle relative URLs
            if href.startswith('/'):
                href = f"{base_wiki_url}{href}"
            elif href.startswith('http'):
                # Keep absolute URLs as is
                pass
            else:
                # Handle other relative cases
                href = f"{base_wiki_url}/wiki/{href}"
            
            # Replace the link with "text (URL)" format
            link.replace_with(f"{text} ({href})")
    
    clean_text = soup.get_text(separator='\n', strip=True)
    
    return title, clean_text, page_data.get('_links', {}).get('webui')

def main():
    # Create data directory if it doesn't exist
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    print(f"✓ Data directory '{data_dir}' ready")
    
    total_articles = 0
    
    for space_idx, space_key in enumerate(space_keys, 1):
        try:
            print(f"\n[{space_idx}/{len(space_keys)}] Processing space: {space_key}")
            print("-" * 50)
            
            all_pages = get_all_pages_in_space(space_key)
            
            if not all_pages:
                print(f"⚠️  No pages found in space '{space_key}'. Skipping.")
                continue

            print(f"📄 Found {len(all_pages)} pages in {space_key}")
            
            # Create knowledge base file for this space
            kb_filename = os.path.join(data_dir, f"{space_key}_kb.txt")
            
            with open(kb_filename, 'w', encoding='utf-8') as f:
                for page_idx, page in enumerate(all_pages, 1):
                    page_id = page['id']
                    page_title = page['title']
                    
                    # Progress indicator
                    print(f"  [{page_idx:3d}/{len(all_pages)}] {page_title[:60]}{'...' if len(page_title) > 60 else ''}")
                    
                    title, content, relative_url = get_page_content(page_id)
                    
                    # Validate content
                    if not content or content.strip() == "":
                        print(f"    ⚠️  Warning: Empty content for '{title}'")
                        content = "[No content available]"
                    
                    # Write in APU knowledge base format
                    f.write(f"--- PAGE: {title} ---\n")
                    if relative_url:
                        f.write(f"URL: {base_wiki_url}{relative_url}\n\n")
                    else:
                        f.write(f"URL: {base_wiki_url}/spaces/{space_key}\n\n")
                    f.write(f"{content}\n\n")
                    
                    total_articles += 1
            
            # File verification
            file_size = os.path.getsize(kb_filename)
            print(f"✓ Saved to '{kb_filename}' ({file_size:,} bytes)")
            
            # Show preview of first article
            with open(kb_filename, 'r', encoding='utf-8') as f:
                preview = f.read(200)
                print(f"📖 Preview: {preview}...")

        except requests.exceptions.HTTPError as err:
            print(f"❌ HTTP Error for space '{space_key}': {err}")
            print("   Please double-check your username, API token, and space key.")
        except Exception as e:
            print(f"❌ Error processing '{space_key}': {e}")
    
    print(f"\n{'='*60}")
    print(f"🎉 Scraping Complete!")
    print(f"📊 Total articles scraped: {total_articles}")
    print(f"📁 Files saved to '{data_dir}/' directory:")
    
    # List created files with sizes
    for space_key in space_keys:
        kb_file = os.path.join(data_dir, f"{space_key}_kb.txt")
        if os.path.exists(kb_file):
            size = os.path.getsize(kb_file)
            print(f"   ✓ {kb_file} ({size:,} bytes)")
        else:
            print(f"   ❌ {kb_file} (not created)")
    
    print(f"\n🤖 Next steps:")
    print(f"   1. Review the files in '{data_dir}/' directory")
    print(f"   2. Run: python main.py")
    print(f"   3. Use 'reindex' command to rebuild vector store")
    print(f"   4. Test chatbot with new knowledge base!")

if __name__ == "__main__":
    main()