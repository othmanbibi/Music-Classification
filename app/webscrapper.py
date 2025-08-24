import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

def get_midi_links(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        midi_links = []
        for a_tag in soup.find_all("a", href=True):
            href = a_tag['href']
            if href.lower().endswith(".mid") or ".midi" in href.lower():
                if href.startswith("http"):
                    midi_links.append(href)
                else:
                    midi_links.append(urljoin(url, href))
        return midi_links
    except Exception as e:
        print(f"Error accessing {url}: {e}")
        return []

# Different URLs to scrap for Meta and Classical Midi Files
genres = {
    "metal": {
        "urls": [
            "https://www.midiworld.com/files/1076/"
        ],
        "folder": "data/midi/metal"
    },
    "classical": {
        "urls": [
            ],
        "folder": "data/midi/classical"
    }
}

# Create folders if not exist
for genre, info in genres.items():
    os.makedirs(info["folder"], exist_ok=True)

# Loop through genres and websites
for genre, info in genres.items():
    print(f"\n===== Scraping {genre.capitalize()} MIDI files =====")
    download_folder = info["folder"]
    
    for url in info["urls"]:
        print(f"Scraping from {url} ...")
        midi_links = get_midi_links(url)
        if not midi_links:
            print(f"No MIDI links found on {url}")
            continue
        
        for link in midi_links:
            filename = os.path.join(download_folder, os.path.basename(link))
            try:
                r = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                r.raise_for_status()
                with open(filename, "wb") as f:
                    f.write(r.content)
                print(f" Downloaded {filename}")
            except Exception as e:
                print(f" Failed to download {link}: {e}")
