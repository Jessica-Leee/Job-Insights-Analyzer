from linkedin_scraper.job_search import JobSearch
import linkedin_scraper.actions as actions
from selenium import webdriver
import csv
from time import sleep
from selenium.webdriver.common.by import By
from typing import List
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service


def save_dict_list_to_csv(data, filename, encoding='utf-8'):
    """
    Save a list of dictionaries to a CSV file.
    
    Args:
        data (list): List of dictionaries where each dictionary represents a row
        filename (str): Path to the output CSV file
        encoding (str, optional): File encoding. Defaults to 'utf-8'.
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not data or not isinstance(data, list):
        print("Error: Data must be a non-empty list of dictionaries")
        return False
    
    try:
        # Get fieldnames from the keys of the first dictionary
        fieldnames = data[0].keys()
        
        with open(filename, 'w', newline='', encoding=encoding) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            writer.writerows(data)
        
        print(f"Successfully saved {len(data)} rows to {filename}")
        return True
    
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        return False


# Create a Service object
service = Service(ChromeDriverManager().install())

# Create the driver with the service
driver = webdriver.Chrome(service=service)
# driver = webdriver.Chrome()
email = "jessica.kexin.le@gmail.com"
password = "Abcd1234!"
print("Logging into LinkedIn...")
actions.login(driver, email, password)
print("Login successful")
job_search = JobSearch(driver=driver, close_on_complete=False, scrape=False)
# job_search contains jobs from your logged in front page:
# - job_search.recommended_jobs
# - job_search.still_hiring
# - job_search.more_jobs

result = []
count = 0
job_listings = job_search.search_all_pages("Manufacturing Engineer in semiconductor", max_pages=30) # returns the list of `Job` from the first page
print(len(job_listings))

print(f"Found {len(job_listings)} job listings")

# Scrape each job and add to results
for job in job_listings:
    try:
        job.scrape()
        result.append(job.to_dict())
        count += 1
        # Print progress every 10 jobs
        if count % 10 == 0:
            print(f"Scraped {count} jobs so far...")
    except Exception as e:
        print(f"Error scraping job: {e}")
    
    # Add a small delay to avoid being rate-limited
    sleep(1)

# Save the final results to CSV
save_dict_list_to_csv(result, "manufacture_jobs_final.csv")
print(f"Completed scraping {count} jobs. Results saved to semiconductor_jobs_final.csv")
