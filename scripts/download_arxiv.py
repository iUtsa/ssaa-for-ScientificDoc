import argparse
import time
import requests
import urllib3
import xml.etree.ElementTree as ET
import os
import re

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def clean_text(text):
    text = re.sub(r'\$[^$]+\$', ' [MATH] ', text)
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def fetch_arxiv_batch(start, max_results=100, category='cs.CL'):
    url = f'http://export.arxiv.org/api/query?search_query=cat:{category}&start={start}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    try:
        r = requests.get(url, timeout=30, verify=False)
        return parse_response(r.text)
    except Exception as e:
        print(f"  Error: {e}")
        return []

def parse_response(data):
    root = ET.fromstring(data)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns)
        abstract = entry.find('atom:summary', ns)
        if title is not None and abstract is not None:
            papers.append({'title': clean_text(title.text or ''), 'abstract': clean_text(abstract.text or '')})
    return papers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_papers', type=int, default=1000)
    parser.add_argument('--output', type=str, default='data/arxiv_corpus.txt')
    parser.add_argument('--categories', nargs='+', default=['cs.CL', 'cs.LG', 'cs.AI', 'cs.CV'])
    args = parser.parse_args()
    print("="*60)
    print("NOVA-SLM v2: arXiv Corpus Downloader")
    print("="*60)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    all_papers = []
    papers_per_cat = args.num_papers // len(args.categories)
    for cat in args.categories:
        print(f"\nDownloading from {cat}...")
        cat_papers, start = [], 0
        while len(cat_papers) < papers_per_cat:
            print(f"  Fetching {start}-{start+100}...")
            batch = fetch_arxiv_batch(start, 100, cat)
            if not batch:
                break
            cat_papers.extend(batch)
            print(f"  Got {len(batch)} (total: {len(cat_papers)})")
            start += 100
            time.sleep(3)
        all_papers.extend(cat_papers[:papers_per_cat])
    with open(args.output, 'w', encoding='utf-8') as f:
        for p in all_papers:
            f.write(f"<abstract> {p['title']}. {p['abstract']}\n")
    print(f"\nTotal: {len(all_papers)} papers saved to {args.output}")

if __name__ == '__main__':
    main()
