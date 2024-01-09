#urls_to_scrape = ["https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/eligibility.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/apply.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/after-apply.html"]

import requests
from bs4 import BeautifulSoup
import csv

def get_text_from_website(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        text_data = soup.get_text()
        return text_data
    else:
        print(f"Error: Unable to fetch the content from {url}. Status code: {response.status_code}")
        return None

def write_to_csv(data, csv_filename):
    # Remove extra spaces and merge the text
    cleaned_data = ' '.join(data.split())
    
    with open(csv_filename, 'a', encoding='utf-8', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([cleaned_data])

# List of URLs to scrape
urls_to_scrape = ["https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/eligibility.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/apply.html", "https://www.canada.ca/en/immigration-refugees-citizenship/services/study-canada/study-permit/student-direct-stream/after-apply.html"]


# CSV filename
csv_filename = "output_data.csv"

# Iterate through each URL, scrape text, and write to CSV
for url in urls_to_scrape:
    text_data = get_text_from_website(url)
    
    if text_data:
        # Write the cleaned and merged text data to CSV, enclosing it within double quotes
        write_to_csv(text_data, csv_filename)
