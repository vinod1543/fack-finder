from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()


mongo_uri = os.getenv("MONGODB_URL")


client = MongoClient(mongo_uri)
db = client["newsDB"]
collection = db["legit_sources"]


legit_websites = [
    {"name": "BBC News", "url": "https://www.bbc.com"},
    {"name": "CNN", "url": "https://www.cnn.com"},
    {"name": "The Hindu", "url": "https://www.thehindu.com"},
    {"name": "The Times of India", "url": "https://timesofindia.indiatimes.com"},
    {"name": "NDTV", "url": "https://www.ndtv.com"},
    {"name": "Reuters", "url": "https://www.reuters.com"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com"},
    {"name": "India Today", "url": "https://www.indiatoday.in"},
    {"name": "Hindustan Times", "url": "https://www.hindustantimes.com"},
    {"name": "The Guardian", "url": "https://www.theguardian.com"},
     { "name": "Reuters", "url": "https://www.reuters.com" },
  { "name": "Associated Press (AP)", "url": "https://apnews.com" },
  { "name": "BBC News", "url": "https://www.bbc.com/news" },
  { "name": "NPR (National Public Radio)", "url": "https://www.npr.org" },
  { "name": "The Guardian", "url": "https://www.theguardian.com" },
  { "name": "Al Jazeera English", "url": "https://www.aljazeera.com" },
  { "name": "Bloomberg", "url": "https://www.bloomberg.com" },
  { "name": "The New York Times", "url": "https://www.nytimes.com" },
  { "name": "The Washington Post", "url": "https://www.washingtonpost.com" },
  { "name": "The Wall Street Journal (WSJ)", "url": "https://www.wsj.com" },
  { "name": "Deutsche Welle (DW)", "url": "https://www.dw.com" },
  { "name": "France 24", "url": "https://www.france24.com" },
  { "name": "Sky News", "url": "https://news.sky.com" },
  { "name": "CBC News (Canada)", "url": "https://www.cbc.ca/news" },
  { "name": "ABC News (Australia)", "url": "https://www.abc.net.au/news" },
  { "name": "Channel News Asia (CNA)", "url": "https://www.channelnewsasia.com" },
  { "name": "Axios", "url": "https://www.axios.com" },
  { "name": "VOA News (Voice of America)", "url": "https://www.voanews.com" },
  { "name": "Politico", "url": "https://www.politico.com" },
  { "name": "TIME", "url": "https://time.com" },
    { "name": "Financial Times", "url": "https://www.ft.com" },
  { "name": "The Economist", "url": "https://www.economist.com" },
  { "name": "NBC News", "url": "https://www.nbcnews.com" },
  { "name": "CBS News", "url": "https://www.cbsnews.com" },
  { "name": "ABC News (USA)", "url": "https://abcnews.go.com" },
  { "name": "Newsweek", "url": "https://www.newsweek.com" },
  { "name": "USA Today", "url": "https://www.usatoday.com" },
  { "name": "The Hill", "url": "https://thehill.com" },
  { "name": "MSNBC", "url": "https://www.msnbc.com" },
  { "name": "CNBC", "url": "https://www.cnbc.com" },
  { "name": "VOX", "url": "https://www.vox.com" },
  { "name": "Los Angeles Times", "url": "https://www.latimes.com" },
  { "name": "Chicago Tribune", "url": "https://www.chicagotribune.com" },
  { "name": "South China Morning Post", "url": "https://www.scmp.com" },
  { "name": "The Times (UK)", "url": "https://www.thetimes.co.uk" },
  { "name": "Le Monde", "url": "https://www.lemonde.fr" },
  { "name": "El País", "url": "https://english.elpais.com" },
  { "name": "RTÉ News (Ireland)", "url": "https://www.rte.ie/news" },
  { "name": "The Telegraph", "url": "https://www.telegraph.co.uk" },
  { "name": "ITV News", "url": "https://www.itv.com/news" },
     { "name": "The Independent", "url": "https://www.independent.co.uk" },
  { "name": "Evening Standard", "url": "https://www.standard.co.uk" },
  { "name": "The Atlantic", "url": "https://www.theatlantic.com" },
  { "name": "Slate", "url": "https://slate.com" },
  { "name": "The New Yorker", "url": "https://www.newyorker.com" },
  { "name": "Foreign Policy", "url": "https://foreignpolicy.com" },
  { "name": "Der Spiegel", "url": "https://www.spiegel.de/international" },
  { "name": "Haaretz", "url": "https://www.haaretz.com" },
  { "name": "The Japan Times", "url": "https://www.japantimes.co.jp" },
  { "name": "Asahi Shimbun", "url": "https://www.asahi.com/ajw/" },
  { "name": "Hindustan Times (Global Edition)", "url": "https://www.hindustantimes.com/world-news" },
  { "name": "NHK World Japan", "url": "https://www3.nhk.or.jp/nhkworld/" },
  { "name": "Jakarta Post", "url": "https://www.thejakartapost.com" },
  { "name": "Korea Herald", "url": "https://www.koreaherald.com" },
  { "name": "Arab News", "url": "https://www.arabnews.com" },
  { "name": "Gulf News", "url": "https://gulfnews.com" },
  { "name": "Straits Times", "url": "https://www.straitstimes.com" },
  { "name": "Bangkok Post", "url": "https://www.bangkokpost.com" },
  { "name": "Mail & Guardian (South Africa)", "url": "https://mg.co.za" },
  { "name": "Daily Maverick", "url": "https://www.dailymaverick.co.za" },
      {"name": "The Hindu", "website": "https://www.thehindu.com/"},
    {"name": "Indian Express", "website": "https://indianexpress.com/"},
    {"name": "LiveMint", "website": "https://www.livemint.com/"},
    {"name": "Business Standard", "website": "https://www.business-standard.com/"},
    {"name": "Scroll.in", "website": "https://scroll.in/"},
    {"name": "The Print", "website": "https://theprint.in/"},
    {"name": "FactChecker.in", "website": "https://www.factchecker.in/"},
    {"name": "Alt News", "website": "https://www.altnews.in/"},
    {"name": "The Wire", "website": "https://thewire.in/"},
    {"name": "The Ken", "website": "https://the-ken.com/"},
    {"name": "Moneycontrol", "website": "https://www.moneycontrol.com/"},
    {"name": "Economic Times", "website": "https://economictimes.indiatimes.com/"},
    {"name": "Outlook India", "website": "https://www.outlookindia.com/"},
    {"name": "The Quint", "website": "https://www.thequint.com/"},
    {"name": "IndiaSpend", "website": "https://www.indiaspend.com/"},
    {"name": "Gaon Connection", "website": "https://en.gaonconnection.com/"},
    {"name": "Newslaundry", "website": "https://www.newslaundry.com/"},
    {"name": "BloombergQuint", "website": "https://www.bqprime.com/"},
    {"name": "Bar and Bench", "website": "https://www.barandbench.com/"},
    {"name": "PIB India", "website": "https://pib.gov.in/"}
]

# Clean up legit_websites: standardize key to 'url', remove non-breaking spaces, and deduplicate
cleaned_sites = []
seen_urls = set()
for site in legit_websites:
    # Standardize key: if 'website' exists, rename to 'url'
    url = site.get('url') or site.get('website')
    if url:
        # Remove non-breaking spaces and strip
        url = url.replace('\u00A0', '').replace('\xa0', '').strip()
        if url not in seen_urls:
            cleaned_sites.append({"name": site["name"], "url": url})
            seen_urls.add(url)

# Insert only unique, cleaned sites into MongoDB
for site in cleaned_sites:
    if not collection.find_one({"url": site["url"]}):
        collection.insert_one(site)

print("Legitimate news websites inserted successfully.")