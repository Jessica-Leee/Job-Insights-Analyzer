# LinkedIn Job Analysis Project

This project analyzes job postings from LinkedIn to gain insights into the job market, particularly focusing on semiconductor and design engineering positions.

## Project Structure

```
.
├── src/
│   ├── scrapers/        # LinkedIn scraping scripts
│   │   ├── main.py     # Main scraping script
│   │   ├── job_search.py
│   │   ├── jobs.py
│   │   ├── person.py
│   │   ├── company.py
│   │   ├── actions.py
│   │   ├── constants.py
│   │   ├── objects.py
│   │   └── selectors.py
│   ├── analysis/        # Data analysis and processing scripts
│   │   └── pre-process.py
│   └── data/           # Data storage directory
├── docs/               # Documentation and presentations
├── tests/             # Test directory
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Setup Instructions

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Scrape LinkedIn data:
   ```bash
   python src/scrapers/main.py
   ```

2. Analyze the data:
   ```bash
   python src/analysis/pre-process.py
   ```

## Components

- **Scrapers**: A comprehensive LinkedIn scraping package that can:
  - Search for jobs
  - Extract job details
  - Get company information
  - Retrieve person profiles

- **Analysis**: Tools for processing and analyzing the collected data

## Data Analysis Results

The analysis results and insights can be found in the `docs` directory, including the presentation file.

## Contributing

Feel free to submit issues and enhancement requests. 