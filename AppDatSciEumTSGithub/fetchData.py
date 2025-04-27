import time
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from io import BytesIO

def fetchData_year(target_year="2022"):
    page_url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    response = requests.get(page_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    #regex pattern to match yellow taxi Parquet files for the target year
    pattern = re.compile(
        rf'https://.*trip-data/yellow_tripdata_({target_year})-\d{{2}}\.parquet'
    )

    #find all <a> tags that have an href attribute and clean URL with .strip()
    links = soup.find_all("a", href=True)
    yearly_links = [link['href'].strip() for link in links if pattern.match(link['href'])]
    yearly_links.sort()

    print(f"Found {len(yearly_links)} Parquet file links for {target_year}.")

    dfs = []
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://www.tlc.nyc.gov'
    }
    
    for file_url in yearly_links:
        print(f"Downloading: {file_url}")
        file_response = requests.get(file_url, headers=headers)
        time.sleep(1)
        file_response.raise_for_status()
        time.sleep(1)
        parquet_stream = BytesIO(file_response.content)
        df = pd.read_parquet(parquet_stream)
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    output_filename = f"yellow_taxi_tripdata_{target_year}_combined.parquet"
    combined_df.to_parquet(output_filename)
    print(f"Combined file saved as {output_filename}")

if __name__ == "__main__":
    target_year = "2024"
    fetchData_year(target_year)
