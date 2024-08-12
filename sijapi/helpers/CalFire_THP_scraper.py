import requests
import PyPDF2
import io
import re

def scrape_data_from_pdf(url):
    response = requests.get(url)
    pdf_file = io.BytesIO(response.content)
    
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    all_text = ""
    for page in pdf_reader.pages:
        all_text += page.extract_text() + "\n"
    
    return all_text

def parse_data(raw_data):
    lines = raw_data.split('\n')
    data = []
    current_entry = None
    
    for line in lines:
        line = line.strip()
        if re.match(r'\d+-\d+-\d+-\w+', line):
            if current_entry:
                data.append(current_entry)
            current_entry = {'Harvest Document': line, 'Raw Data': []}
        elif current_entry:
            current_entry['Raw Data'].append(line)
    
    if current_entry:
        data.append(current_entry)
    
    return data

def filter_data(data):
    return [entry for entry in data if any(owner.lower() in ' '.join(entry['Raw Data']).lower() for owner in ["Sierra Pacific", "SPI", "Land & Timber"])]

def extract_location(raw_data):
    location = []
    for line in raw_data:
        if 'MDBM:' in line or 'HBM:' in line:
            location.append(line)
    return ' '.join(location)

def extract_plss_coordinates(text):
    pattern = r'(\w+): T(\d+)([NSEW]) R(\d+)([NSEW]) S(\d+)'
    return re.findall(pattern, text)

# Main execution
url = "https://caltreesplans.resources.ca.gov/Caltrees/Report/ShowReport.aspx?module=TH_Document&reportID=492&reportType=LINK_REPORT_LIST"
raw_data = scrape_data_from_pdf(url)

parsed_data = parse_data(raw_data)
print(f"Total timber plans parsed: {len(parsed_data)}")

filtered_data = filter_data(parsed_data)
print(f"Found {len(filtered_data)} matching entries.")

for plan in filtered_data:
    print("\nHarvest Document:", plan['Harvest Document'])
    
    location = extract_location(plan['Raw Data'])
    print("Location:", location)
    
    plss_coordinates = extract_plss_coordinates(location)
    print("PLSS Coordinates:")
    for coord in plss_coordinates:
        meridian, township, township_dir, range_, range_dir, section = coord
        print(f"  {meridian}: T{township}{township_dir} R{range_}{range_dir} S{section}")
    
    print("-" * 50)
