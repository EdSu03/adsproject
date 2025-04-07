from DOPU_given_timerange import get_cleaned_df
from Scrape_Data import download_and_read_parquet_files
from data_cleaning import clean_parquet_files_yellow_taxi, clean_parquet_files_green_taxi


def main(url):

    download_and_read_parquet_files(url, taxi_type = "yellow")
    download_and_read_parquet_files(url, taxi_type = "green")

    clean_parquet_files_yellow_taxi()
    clean_parquet_files_green_taxi()

    get_cleaned_df()


def __main__(*args):
    url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
    main(url)
