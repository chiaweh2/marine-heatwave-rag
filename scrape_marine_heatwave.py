#!/usr/bin/env python3
"""
Scrape marine heatwave discussion from NOAA PSL website
This script extracts forecast discussion sections for RAG processing
"""
import requests
import re
from pathlib import Path
from typing import List, Dict, Optional
import json
from datetime import datetime
from bs4 import BeautifulSoup

# Configuration
NOAA_PSL_URL = "https://psl.noaa.gov/marine-heatwaves/#report"
DATA_DIR = Path(__file__).parent / "data"
SYNC_LOG = DATA_DIR / "sync_log.json"

def ensure_data_dir():
    """Ensure data directory exists"""
    if not DATA_DIR.exists():
        DATA_DIR.mkdir()

def load_sync_log() -> List:
    """Load previous sync log as a list of entries"""
    if SYNC_LOG.exists():
        try:
            with open(SYNC_LOG, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []

def add_to_sync_log(forecast_date: str, status: str):
    """Add new entry to sync log at the top"""
    log_entries = load_sync_log()

    # Create new entry
    new_entry = {
        "timestamp": datetime.now().isoformat(),
        "forecast_date": forecast_date,
        "status": status,
        "source_url": NOAA_PSL_URL
    }

    # Add to the beginning (newest first)
    log_entries.insert(0, new_entry)

    # Keep only the last 50 entries to prevent file from growing too large
    log_entries = log_entries[:50]

    # Save back to file
    with open(SYNC_LOG, 'w', encoding='utf-8') as f:
        json.dump(log_entries, f, indent=2)

def fetch_webpage() -> Optional[BeautifulSoup]:
    """Fetch and parse the NOAA PSL marine heatwave page"""
    try:
        print(f"🌐 Fetching {NOAA_PSL_URL}...")
        headers = {
            'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
            )
        }
        try:
            response = requests.get(NOAA_PSL_URL, headers=headers, timeout=60)
        except requests.Timeout:
            print("✗ Request timed out after 60 seconds. Aborting.")
            return None

        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        print("✓ Successfully fetched webpage")
        return soup

    except requests.RequestException as e:
        print(f"✗ Error fetching webpage: {e}")
        return None

def convert_html_to_markdown(element) -> str:
    """Convert HTML element to Markdown format"""
    if element.name == 'h3':
        return f"### {element.get_text().strip()}\n\n"
    elif element.name == 'h4':
        return f"#### {element.get_text().strip()}\n\n"
    elif element.name == 'h5':
        return f"##### {element.get_text().strip()}\n\n"
    elif element.name == 'p':
        text = element.get_text().strip()
        if text:
            return f"{text}\n\n"
        return ""
    elif element.name == 'li':
        text = element.get_text().strip()
        if text:
            return f"- {text}\n"
        return ""
    elif element.name == 'ul' or element.name == 'ol':
        # Handle lists by processing their li children
        items = []
        for li in element.find_all('li', recursive=False):
            item_text = li.get_text().strip()
            if item_text:
                items.append(f"- {item_text}")
        if items:
            return '\n'.join(items) + '\n\n'
        return ""
    else:
        # For other elements, just extract text
        text = element.get_text().strip()
        if text:
            return f"{text}\n\n"
        return ""

def extract_discussion_sections(soup: BeautifulSoup) -> Optional[Dict]:
    """Extract the specific discussion sections from the webpage"""
    try:
        # Find all h5 tags and search for the specific ones we need
        all_h5_tags = soup.find_all('h5')
        forecast_initial_h5 = None
        forecast_period_h5 = None

        for h5 in all_h5_tags:
            h5_text = h5.get_text().lower()
            if 'forecast initial time' in h5_text:
                forecast_initial_h5 = h5
            elif 'forecast period' in h5_text:
                forecast_period_h5 = h5

        # Find the Global Marine Heatwave Forecast Discussion h3
        discussion_h3 = soup.find(
            'h3', string=re.compile(r'Global Marine Heatwave Forecast Discussion', re.IGNORECASE)
        )

        if not all([forecast_initial_h5, forecast_period_h5, discussion_h3]):
            print("✗ Could not find all required sections")
            return None

        print("✓ Found all required sections")

        # Extract content and convert to markdown
        markdown_content = []

        # Add forecast initial time text only
        forecast_initial_text = forecast_initial_h5.get_text().strip()
        if forecast_initial_text:
            markdown_content.append(f"##### {forecast_initial_text}\n\n")
        
        # Extract date from the strong tag within forecast initial time
        strong_tag = forecast_initial_h5.find('strong')
        if strong_tag:
            forecast_date = strong_tag.get_text().strip().replace(' ', '_')
        else:
            # Fallback: extract from full text
            match = re.search(r'(\w+ \d{4})', forecast_initial_text)
            forecast_date = match.group(1).replace(' ', '_') if match else 'unknown_date'

        # Add forecast period text only
        forecast_period_text = forecast_period_h5.get_text().strip()
        if forecast_period_text:
            markdown_content.append(f"##### {forecast_period_text}\n\n")
        
        # Extract period from the strong tag within forecast period
        strong_tag_period = forecast_period_h5.find('strong')
        if strong_tag_period:
            forecast_period = strong_tag_period.get_text().strip().replace(' ', '_')
        else:
            # Fallback: extract from full text
            forecast_period = forecast_period_text.replace("Forecast period", "").strip()
            forecast_period = "_".join(forecast_period.split())

        # Collect all content between discussion_h3 and basinDiv
        current = discussion_h3
        while current:
            # Stop if we reach a div with class 'basinDiv'
            if (current.name == 'div' and 
                current.get('class') and 
                'basinDiv' in current.get('class')):
                break

            # Process and convert current element to markdown
            if hasattr(current, 'name') and current.name:
                markdown_text = convert_html_to_markdown(current)
                if markdown_text.strip():
                    markdown_content.append(markdown_text)

            # Traverse to next sibling in tag tree
            current = current.next_sibling

            # Skip text nodes (whitespace between elements)
            while current and current.name is None:
                current = current.next_sibling

        # Create structured data with markdown content
        discussion_data = {
            'timestamp': datetime.now().isoformat(),
            'source_url': NOAA_PSL_URL,
            'forecast_date': forecast_date,
            'forecast_period': forecast_period,
            'markdown_content': ''.join(markdown_content).strip()
        }

        return discussion_data

    except Exception as e:
        print(f"✗ Error extracting discussion sections: {e}")
        return None

def save_discussion_data(discussion_data: Dict) -> bool:
    """Save discussion data as markdown file"""
    try:
        forecast_date = discussion_data.get('forecast_date', 'unknown_date')
        filename = f"marine_heatwave_discussion_init_{forecast_date}.md"
        file_path = DATA_DIR / filename

        # If file exists, stop scraping
        if file_path.exists():
            print(f"✓ Discussion for {forecast_date} already exists: {filename}")
            return False

        # Compose markdown content with metadata header (no weird indentation)
        markdown_content = (
            "---\n"
            "title: Marine Heatwave Forecast Discussion\n"
            f"source: {discussion_data['source_url']}\n"
            f"extracted: {discussion_data['timestamp']}\n"
            "---\n\n"
            "# Marine Heatwave Forecast Discussion\n\n"
            f"**Source:** [{discussion_data['source_url']}]({discussion_data['source_url']})  \n"
            f"**Extracted:** {discussion_data['timestamp']}\n\n"
            "---\n\n"
            f"{discussion_data['markdown_content']}\n"
        )

        # Save markdown file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"✓ Saved discussion data to {filename}")
        return True

    except Exception as e:
        print(f"✗ Error saving discussion data: {e}")
        return False

def scrape_marine_heatwave_discussion():
    """Main function to scrape marine heatwave discussion"""
    ensure_data_dir()

    print("🌊 Starting marine heatwave discussion scraping...")

    # Fetch webpage
    soup = fetch_webpage()
    if not soup:
        return

    # Extract discussion sections
    discussion_data = extract_discussion_sections(soup)
    if not discussion_data:
        return

    # Save the data
    saved_successfully = save_discussion_data(discussion_data)

    # Extract forecast date for logging
    forecast_date = discussion_data.get('forecast_date', 'unknown_date')
    # Add to sync log
    if saved_successfully:
        add_to_sync_log(forecast_date, "success")
        print(f"\n🎉 Successfully scraped marine heatwave discussion!")
        print(f"📁 Data saved in: {DATA_DIR}")
    else:
        add_to_sync_log(forecast_date, "already_exists")
        print(f"\n📋 Discussion for {forecast_date} already exists - skipped scraping")

def list_local_discussions():
    """List locally saved discussions"""
    ensure_data_dir()
    md_files = list(DATA_DIR.glob("marine_heatwave_discussion_*.md"))

    if md_files:
        print("📁 Local marine heatwave discussions:")
        for file_path in sorted(md_files):
            size = file_path.stat().st_size
            print(f"  - {file_path.name} ({size:,} bytes)")
    else:
        print("📁 No local discussions found")

def show_sync_log():
    """Show recent scraping activity"""
    log_entries = load_sync_log()

    if log_entries:
        print("📋 Recent scraping activity (newest first):")
        for entry in log_entries[:10]:  # Show last 10 entries
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            forecast_date = entry.get('forecast_date', 'Unknown')
            status = entry.get('status', 'unknown')
            
            if status == "success":
                status_icon = "✅"
            elif status == "already_exists":
                status_icon = "⏭️"
            else:
                status_icon = "❌"
            print(f"  {status_icon} {timestamp} - {forecast_date} ({status})")
    else:
        print("📋 No scraping activity logged yet")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape marine heatwave discussion from NOAA PSL website"
    )
    parser.add_argument("--list", action="store_true", help="List local discussions only")
    parser.add_argument("--log", action="store_true", help="Show recent scraping activity log")

    args = parser.parse_args()

    if args.list:
        list_local_discussions()
    elif args.log:
        show_sync_log()
    else:
        scrape_marine_heatwave_discussion()
