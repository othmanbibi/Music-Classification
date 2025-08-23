import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# URL of metal genre page
base_url = "https://metal-midi.grahamdowney.com/midi.html"
download_folder = "data/midi/metal"
os.makedirs(download_folder, exist_ok=True)

# Get page content
response = requests.get(base_url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find all MIDI links
links = soup.find_all('a', href=True)
midi_links = [urljoin(base_url, link['href']) for link in links if link['href'].endswith('.mid')]

# Download MIDI files
for url in midi_links:
    filename = os.path.join(download_folder, url.split('/')[-1])
    r = requests.get(url)
    r.raise_for_status()  # optional: raises an error if download fails
    with open(filename, 'wb') as f:
        f.write(r.content)
    print(f"Downloaded {filename}")
