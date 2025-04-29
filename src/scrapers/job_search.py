import os
import time
from typing import List
from time import sleep
import urllib.parse

from .objects import Scraper
from . import constants as c
from .jobs import Job

from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys


class JobSearch(Scraper):
    AREAS = ["recommended_jobs", None, "still_hiring", "more_jobs"]

    def __init__(self, driver, base_url="https://www.linkedin.com/jobs/", close_on_complete=False, scrape=True, scrape_recommended_jobs=True):
        super().__init__()
        self.driver = driver
        self.base_url = base_url

        if scrape:
            self.scrape(close_on_complete, scrape_recommended_jobs)


    def scrape(self, close_on_complete=True, scrape_recommended_jobs=True):
        if self.is_signed_in():
            self.scrape_logged_in(close_on_complete=close_on_complete, scrape_recommended_jobs=scrape_recommended_jobs)
        else:
            raise NotImplemented("This part is not implemented yet")


    def scrape_job_card(self, base_element) -> Job:
        # Find elements within the base_element, not from the driver
        job_div = base_element.find_element(By.CLASS_NAME, "job-card-list__entity-lockup")
        
        # Get job title from the found element
        job_title = job_div.text.strip()
        
        # Get LinkedIn URL
        job_link = base_element.find_element(By.CSS_SELECTOR, 
            ".job-card-container__link, .job-card-list__title--link")
        
        # Extract href from this element
        linkedin_url = job_link.get_attribute("href")
        
        # Use the updated method to find elements within base_element
        company = base_element.find_element(By.CLASS_NAME, "artdeco-entity-lockup__subtitle").text
        
        # Use the updated method for location
        location = base_element.find_element(By.CLASS_NAME, "job-card-container__metadata-wrapper").text
        
        # Create and return the Job object
        job = Job(
            linkedin_url=linkedin_url,
            job_title=job_title,
            company=company,
            location=location,
            scrape=False,
            driver=self.driver
        )
        # print(linkedin_url) debug
        return job


    def scrape_logged_in(self, close_on_complete=True, scrape_recommended_jobs=True):
        driver = self.driver
        driver.get(self.base_url)
        if scrape_recommended_jobs:
            self.focus()
            sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)
            job_area = self.wait_for_element_to_load(name="scaffold-finite-scroll__content")
            areas = self.wait_for_all_elements_to_load(name="artdeco-card", base=job_area)
            for i, area in enumerate(areas):
                area_name = self.AREAS[i]
                if not area_name:
                    continue
                area_results = []
                for job_posting in area.find_elements_by_class_name("jobs-job-board-list__item"):
                    job = self.scrape_job_card(job_posting)
                    area_results.append(job)
                setattr(self, area_name, area_results)
        return

    def search(self, search_term: str) -> List[Job]:
        url = os.path.join(self.base_url, "search") + f"?keywords={urllib.parse.quote(search_term)}&refresh=true"
        self.driver.get(url)
        self.scroll_to_bottom()
        self.focus()
        sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)

        job_listing_class_name = "FYNeDiftQySlLPrJMPPTjHldfSVjBNPaSQ"
        # job_listing = self.wait_for_element_to_load(name=job_listing_class_name)
        # job_listing = self.wait_for_element_to_load(class_name=job_listing_class_name)
        job_listing = self.driver.find_element(By.CLASS_NAME, job_listing_class_name)

        for i in range(1, 11):
            self.scroll_class_name_element_to_page_percent(job_listing_class_name, i / 10)
            self.focus()
            sleep(self.WAIT_FOR_ELEMENT_TIMEOUT)


        job_results = []
        job_cards = self.wait_for_all_elements_to_load(name="job-card-list", base=job_listing)
        print(f"✅ Found {len(job_cards)} job cards.")
        for job_card in job_cards:
            job = self.scrape_job_card(job_card)
            job_results.append(job)
        return job_results
    

    def wait_for_pagination_to_stabilize(self, timeout=30, check_interval=2):
        """
        Wait for pagination to fully load and stabilize by checking if the number of page buttons
        remains constant for a certain period.
        
        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds
            
        Returns:
            Boolean indicating if pagination has stabilized
        """
        print("Waiting for pagination to stabilize...")
        start_time = time.time()
        previous_button_count = 0
        stable_count = 0
        required_stable_checks = 3  # Number of consistent checks required to consider it stable
        
        while time.time() - start_time < timeout:
            try:
                # Try to find pagination area
                pagination_area = self.driver.find_element(By.CLASS_NAME, "jobs-search-pagination")
                page_buttons = pagination_area.find_elements(By.CLASS_NAME, "jobs-search-pagination__indicator-button")
                current_button_count = len(page_buttons)
                
                # Check if the count has remained stable
                if current_button_count == previous_button_count and current_button_count > 0:
                    stable_count += 1
                    if stable_count >= required_stable_checks:
                        print(f"Pagination stabilized with {current_button_count} page buttons")
                        return True
                else:
                    stable_count = 0
                    
                previous_button_count = current_button_count
                print(f"Found {current_button_count} pagination buttons, waiting for stability...")
            except Exception as e:
                print(f"Pagination not yet available: {str(e)}")
                
            sleep(check_interval)
            
        print("Pagination failed to stabilize within timeout")
        return False
    
    def search_all_pages(self, search_term: str, max_pages: int = 5) -> List[Job]:
        """
        Search for jobs across multiple pages and scrape all results.
        
        Args:
            search_term: The keyword to search for
            max_pages: Maximum number of pages to scrape (default: 5)
            
        Returns:
            List of Job objects from all pages
        """
        # Time module is now imported at the top of the file
        
        all_jobs = []
        
        # Get initial search results
        first_page_jobs = self.search(search_term)
        all_jobs.extend(first_page_jobs)
        print("Scraped initial page")
        
        # Ensure pagination has fully loaded and stabilized before proceeding
        # Wait longer for the initial page load to complete
        sleep(3)  # Short initial wait
        
        # Scroll to the bottom to trigger loading of all pagination elements
        self.scroll_to_bottom()
        sleep(2)
        
        # Wait for the pagination to stabilize
        pagination_ready = self.wait_for_pagination_to_stabilize()
        if not pagination_ready:
            print("Warning: Pagination may not be fully loaded. Proceeding anyway.")
        
        current_page = 1
        while current_page < max_pages:
            try:
                # Try to find the pagination area
                try:
                    pagination_area = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "jobs-search-pagination"))
                    )
                except Exception as e:
                    print(f"Pagination area not found after waiting: {str(e)}")
                    print("No more pages available or pagination not loaded")
                    break
                
                # Find all page buttons to determine if there are more pages
                page_buttons = pagination_area.find_elements(By.CLASS_NAME, "jobs-search-pagination__indicator-button")
                
                if not page_buttons:
                    print("No pagination buttons found")
                    break
                    
                # Find the button for the next page
                next_page_number = current_page + 1
                next_page_button = None
                
                for button in page_buttons:
                    button_text = button.text.strip()
                    if button_text and button_text.isdigit() and int(button_text) == next_page_number:
                        next_page_button = button
                        break
                
                # If no next page button found, we've reached the end
                if not next_page_button:
                    print(f"No button found for page {next_page_number}")
                    break
                
                print(f"Clicking button for page {next_page_number}")
                # Use JavaScript to click the button to avoid any overlay issues
                self.driver.execute_script("arguments[0].click();", next_page_button)
                
                # Wait for the page to load
                sleep(self.WAIT_FOR_ELEMENT_TIMEOUT * 2)  # Double the wait time for page transitions
                
                # Verify we're on the new page by checking the URL or a page indicator
                try:
                    # Ensure the page has changed - this could be checked via URL parameters
                    # or by verifying a page indicator element
                    WebDriverWait(self.driver, 10).until(
                        EC.staleness_of(next_page_button)
                    )
                    print(f"Successfully navigated to page {next_page_number}")
                except:
                    print("Page navigation may have failed, continuing anyway")
                
                # Full scroll sequence to ensure all content is loaded
                self.scroll_to_bottom()
                sleep(1)
                
                job_listing_class_name = "FYNeDiftQySlLPrJMPPTjHldfSVjBNPaSQ"
                
                try:
                    job_listing = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, job_listing_class_name))
                    )
                except Exception as e:
                    print(f"Job listing container not found on page {next_page_number}: {str(e)}")
                    job_listing = None
                
                if job_listing:
                    for i in range(1, 11):
                        try:
                            self.scroll_class_name_element_to_page_percent(job_listing_class_name, i / 10)
                            self.focus()
                            sleep(1)  # Shorter sleep between scrolls
                        except:
                            pass  # Continue if scroll fails
                    
                    # Give additional time for all jobs to load after scrolling
                    sleep(2)
                    
                    # Scrape jobs from this page
                    page_jobs = []
                    try:
                        job_cards = self.wait_for_all_elements_to_load(name="job-card-list", base=job_listing)
                        print(f"Found {len(job_cards)} job cards on page {next_page_number}")
                        
                        for job_card in job_cards:
                            try:
                                job = self.scrape_job_card(job_card)
                                page_jobs.append(job)
                            except Exception as e:
                                print(f"Error scraping a job card: {str(e)}")
                                continue
                            
                        # Add jobs from this page to the total
                        all_jobs.extend(page_jobs)
                        print(f"Scraped {len(page_jobs)} jobs from page {next_page_number}")
                    except Exception as e:
                        print(f"Error finding job cards on page {next_page_number}: {str(e)}")
                
                # Update current page
                current_page = next_page_number
                
            except Exception as e:
                print(f"Error navigating to page {current_page + 1}: {str(e)}")
                # Take a screenshot for debugging
                try:
                    screenshot_path = f"error_page_{current_page + 1}.png"
                    self.driver.save_screenshot(screenshot_path)
                    print(f"Error screenshot saved to {screenshot_path}")
                except:
                    pass
                break
        
        return all_jobs