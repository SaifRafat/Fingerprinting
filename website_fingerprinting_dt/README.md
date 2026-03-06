# Website Fingerprinting Data Collection - YouTube Search Traffic

## Overview

This project collects network traffic data from YouTube searches for website fingerprinting research. The dataset contains 50 PCAP files capturing encrypted TCP traffic during different YouTube search queries.

## Methodology

### Step 1: Create Search Queries
- Created 50 different YouTube search queries
- Stored in `youtube_prompt.xlsx`

### Step 2: Automated Data Collection
- Developed `automated.ipynb` script using Selenium WebDriver
- Automates the entire data collection process to save time
- Instead of manually performing 50 searches, the script does it automatically

### Step 3: Traffic Capture
- Captured only **TCP traffic on port 443** (HTTPS)
- Used packet capture tools to record network packets

### Step 4: Search Process (per query)
For each of the 50 search queries, the script:
1. Searches the query on YouTube
2. Scrolls through the search results page
3. Clicks on the first video result
4. Records all TCP traffic during this process

## Output

- **`youtube_search_[1-50]_capture.pcap`**: 50 PCAP files
  - One file per search query
  - Contains TCP traffic on port 443
  - Ready for fingerprinting analysis

## Files

- `automated.ipynb` - Automated collection script (Selenium)
- `youtube_prompt.xlsx` - 50 search queries
- `youtube_search_1-50_capture.pcap` - Captured traffic data
