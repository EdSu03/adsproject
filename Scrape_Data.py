#!/usr/bin/env python
# coding: utf-8
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from tqdm.notebook import tqdm

import time
from urllib.parse import urljoin


# In[12]:
## Get the data!

# Toggle whether to use the whole data or just a sample
USE_SAMPLE = True

# Scrape the given URL for all parquet file links
def get_parquets(url, taxi_type = "yellow"):
    response = requests.get(url)
    if response.status_code != 200:
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    return [
        a["href"].strip()
        for a in soup.find_all("a", href=True)
        if a["href"].strip().endswith(".parquet")
        and (not USE_SAMPLE or (taxi_type in a["href"] and "2024" in a["href"]))
    ]

# Download a single file while displaying a progress bar
def download_file(url, save_path, position=1):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        position=position,
        leave=False,
        dynamic_ncols=True,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))

# Download and process all parquet files found on the webpage
def download_and_read_parquet_files(url, save_dir="data_test", taxi_type = "yellow"):
    os.makedirs(save_dir, exist_ok=True)
    
    links = get_parquets(url, taxi_type)
    dataframes = []
    
    with tqdm(total=len(links), desc="Processing files", unit="file", position=0, leave=True) as file_bar:
        for link in links:
            full_link = urljoin(url, link)
            filename = os.path.join(save_dir, os.path.basename(link))

            if not os.path.exists(filename):
                download_file(full_link, filename, position=1)
                time.sleep(5)

            attempt = 0
            while attempt < 2:
                try:
                    df = pd.read_parquet(filename, engine='fastparquet')
                    df[df.select_dtypes(include=['number']).columns] = df.select_dtypes(include=['number']).apply(pd.to_numeric, downcast='float')
                    dataframes.append(df)
                    break
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    if attempt == 0:
                        os.remove(filename)
                        print("Redownloading...")
                        download_file(full_link, filename, position=1)
                        time.sleep(5)
                    attempt += 1
            
            file_bar.update(1)

    return dataframes

url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
dfs = download_and_read_parquet_files(url, taxi_type = "yellow")





