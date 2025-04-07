from selenium.common.exceptions import TimeoutException

from .objects import Scraper
from . import constants as c
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class Job(Scraper):

    def __init__(
        self,
        linkedin_url=None,
        job_title=None,
        company=None,
        location=None,
        posted_date=None,
        job_description=None,
        benefits=None,
        driver=None,
        close_on_complete=True,
        scrape=True,
    ):
        super().__init__()
        self.linkedin_url = linkedin_url
        self.job_title = job_title
        self.driver = driver
        self.company = company
        self.location = location
        self.posted_date = posted_date
        self.job_description = job_description

        if scrape:
            self.scrape(close_on_complete)

    def __repr__(self):
        return f"<Job {self.job_title} {self.company}>"

    def scrape(self, close_on_complete=True):
        if self.is_signed_in():
            self.scrape_logged_in(close_on_complete=close_on_complete)
        else:
            raise NotImplemented("This part is not implemented yet")

    def to_dict(self):
        return {
            "linkedin_url": self.linkedin_url,
            "job_title": self.job_title,
            "company": self.company,
            "location": self.location,
            "posted_date": self.posted_date,
            "job_description": self.job_description,
        }
        



    def scrape_logged_in(self, close_on_complete=True):
        driver = self.driver
        driver.get(self.linkedin_url)
        self.focus()
        
        # Use try-except around each element fetch
        try:
            self.job_title = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__job-title").text.strip()
        except Exception as e:
            print(f"Error getting job title: {e}")
            self.job_title = "N/A"
        
        try:
            self.company = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__company-name").text.strip()
        except Exception as e:
            print(f"Error getting company: {e}")
            self.company = "N/A"
        
        # Get location and posted date
        try:
            primary_descriptions = self.wait_for_element_to_load(
                name="job-details-jobs-unified-top-card__primary-description-container"
            ).find_elements(By.TAG_NAME, "span")
            texts = [span.text for span in primary_descriptions if span.text.strip() != ""]
            self.location = texts[0] if len(texts) > 0 else "N/A"
            self.posted_date = texts[3] if len(texts) > 3 else "N/A"
        except Exception as e:
            print(f"Error getting location/date: {e}")
            self.location = "N/A"
            self.posted_date = "N/A"
        
        
        
        # Fix the job description part to handle stale elements
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get job description with retry logic
                job_description_elem = self.wait_for_element_to_load(name="jobs-description")
                
                # Look for the button and click it if it exists
                try:
                    button = WebDriverWait(job_description_elem, 5).until(
                        EC.element_to_be_clickable((By.TAG_NAME, "button"))
                    )
                    button.click()
                    # Wait for content to load after click
                    import time
                    time.sleep(1)
                except:
                    # No button or already expanded
                    pass
                
                # Get the updated description after potential expansion
                job_description_elem = self.wait_for_element_to_load(name="jobs-description")
                self.job_description = job_description_elem.text.strip()
                break  # Success
            except Exception as e:
                print(f"Error getting job description (attempt {attempt+1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    self.job_description = "Failed to retrieve"
        
     


    # def scrape_logged_in(self, close_on_complete=True):
    #     driver = self.driver
        
    #     driver.get(self.linkedin_url)
    #     self.focus()
    #     self.job_title = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__job-title").text.strip()
    #     self.company = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__company-name").text.strip()
    #     # self.company_linkedin_url = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__company-name").find_element(By.TAG_NAME,"a").get_attribute("href")
    #     primary_descriptions = self.wait_for_element_to_load(name="job-details-jobs-unified-top-card__primary-description-container").find_elements(By.TAG_NAME, "span")
    #     texts = [span.text for span in primary_descriptions if span.text.strip() != ""]
    #     self.location = texts[0]
    #     self.posted_date = texts[3]
        
    #     try:
    #         self.applicant_count = self.wait_for_element_to_load(name="jobs-unified-top-card__applicant-count").text.strip()
    #     except TimeoutException:
    #         self.applicant_count = 0
    #     job_description_elem = self.wait_for_element_to_load(name="jobs-description")
    #     self.mouse_click(job_description_elem.find_element(By.TAG_NAME, "button"))
    #     job_description_elem = self.wait_for_element_to_load(name="jobs-description")
    #     job_description_elem.find_element(By.TAG_NAME, "button").click()
    #     self.job_description = job_description_elem.text.strip()
    #     try:
    #         self.benefits = self.wait_for_element_to_load(name="jobs-unified-description__salary-main-rail-card").text.strip()
    #     except TimeoutException:
    #         self.benefits = None

    #     if close_on_complete:
    #         driver.close()
