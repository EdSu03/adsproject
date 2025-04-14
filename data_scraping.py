import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
from tqdm.notebook import tqdm
import io

import time
from urllib.parse import urljoin

## Get the data!

# Toggle whether to use the whole data or just a sample
USE_SAMPLE = True


# Scrape the given URL for all parquet file links
def get_parquets(url, taxi_type="yellow", years=[2024]):
    response = requests.get(url)
    if response.status_code != 200:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    return [
        a["href"].strip()
        for a in soup.find_all("a", href=True)
        if a["href"].strip().endswith(".parquet")
        and (
            not USE_SAMPLE
            or (
                taxi_type in a["href"] and any(str(year) in a["href"] for year in years)
            )
        )
    ]


# Download a single file while displaying a progress bar
def download_and_optimize_parquet(url, save_path, position=1):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    raw_data = io.BytesIO()

    with tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        position=position,
        leave=False,
        dynamic_ncols=True,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                raw_data.write(chunk)
                bar.update(len(chunk))

    raw_data.seek(0)

    # Optimise the DataFrame
    df = pd.read_parquet(raw_data, engine="fastparquet")

    for col in df.select_dtypes(include=["number"]).columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].astype("float32")

    df.to_parquet(save_path, engine="fastparquet", index=False)

    return df


def download_and_read_parquet_files(
    url, save_dir="data", taxi_type="yellow", years=[2024]
):
    os.makedirs(save_dir, exist_ok=True)

    links = get_parquets(url, taxi_type, years)
    dataframes = []

    with tqdm(
        total=len(links), desc="Processing files", unit="file", position=0, leave=True
    ) as file_bar:
        for link in links:
            full_link = urljoin(url, link)
            filename = os.path.join(save_dir, os.path.basename(link))

            if not os.path.exists(filename):
                try:
                    df = download_and_optimize_parquet(full_link, filename, position=1)
                    dataframes.append(df)
                    file_bar.update(1)
                    time.sleep(5)
                    continue
                except Exception as e:
                    print(f"Error downloading/processing {filename}: {e}")
                    continue

            # Fallback if file exists already
            attempt = 0
            while attempt < 2:
                try:
                    df = pd.read_parquet(filename, engine="fastparquet")
                    dataframes.append(df)
                    break
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    if attempt == 0:
                        os.remove(filename)
                        print("Redownloading...")
                        try:
                            df = download_and_optimize_parquet(
                                full_link, filename, position=1
                            )
                            dataframes.append(df)
                            break
                        except Exception as e2:
                            print(f"Retry failed: {e2}")
                    attempt += 1

            file_bar.update(1)

    return dataframes


url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
dfs = download_and_read_parquet_files(url, taxi_type="yellow", years=[2024, 2023])
