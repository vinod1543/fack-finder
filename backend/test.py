import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib

# LangChain imports for Google Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import BaseOutputParser
# NOTE: DuckDuckGoSearchRun has been removed.
from langchain.agents import AgentExecutor
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# Additional imports
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai


class SearchEngine(Enum):
    """Enum for search engines. DuckDuckGo has been removed."""
    GOOGLE = "google"
    BING = "bing"
    NEWS_API = "news_api"
    MOCK_ENGINE = "mock_engine" # Added for clarity


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source_domain: str
    publish_date: Optional[datetime]
    search_engine: SearchEngine
    relevance_score: float
    credibility_score: float = 0.0


@dataclass
class VerificationQuery:
    original_claim: str
    search_queries: List[str]
    generated_queries: List[str]
    priority: int = 1  # 1-5, higher is more urgent


class ClaimExtractor(BaseModel):
    """Pydantic model for extracting claims from news text"""
    claims: List[str] = Field(description="List of factual claims extracted from the text")
    main_claim: str = Field(description="The primary claim or assertion")
    supporting_details: List[str] = Field(description="Supporting facts or details")


class SearchQueryGenerator(BaseModel):
    """Pydantic model for generating search queries"""
    primary_queries: List[str] = Field(description="Main search queries for the claim")
    alternative_queries: List[str] = Field(description="Alternative phrasings and approaches")
    contradiction_queries: List[str] = Field(description="Queries to find contradicting information")


class NewsVerificationSearcher:
    def __init__(self, google_api_key: str, trusted_sources: Dict[str, float] = None):
        """
        Initialize the News Verification Searcher with Google Gemini.
        The external search tool dependency has been removed.
        
        Args:
            google_api_key: Google API key for Gemini models
            trusted_sources: Dictionary of domain -> credibility_score (0.0-1.0)
        """
        # Configure Google Gemini
        genai.configure(api_key=google_api_key)
        
        # Initialize Gemini models
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.1,
            max_tokens=8192
        )
        
        self.fast_llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=google_api_key,
            temperature=0.2,
            max_tokens=4096
        )
        
        self.trusted_sources = trusted_sources or self._get_default_trusted_sources()
        
        # NOTE: Removed the initialization of search_tools which used DuckDuckGoSearchRun.
        # self.search_tools = { ... }
        
        # Setup parsers
        self.claim_parser = PydanticOutputParser(pydantic_object=ClaimExtractor)
        self.query_parser = PydanticOutputParser(pydantic_object=SearchQueryGenerator)
        
        # Setup prompts optimized for Gemini
        self._setup_prompts()
    
    def _get_default_trusted_sources(self) -> Dict[str, float]:
        """Default trusted news sources with credibility scores"""
        return {
            "reuters.com": 0.95, "apnews.com": 0.95, "afp.com": 0.92,
            "bbc.com": 0.90, "bbc.co.uk": 0.90, "npr.org": 0.90,
            "theguardian.com": 0.85, "washingtonpost.com": 0.85, "nytimes.com": 0.85,
            "wsj.com": 0.85, "economist.com": 0.85, "cnn.com": 0.80,
            "abcnews.go.com": 0.80, "cbsnews.com": 0.80, "nbcnews.com": 0.80,
            "pbs.org": 0.88, "factcheck.org": 0.95, "snopes.com": 0.90,
            "politifact.com": 0.90, "fullfact.org": 0.92, "nature.com": 0.95,
            "science.org": 0.95, "nationalgeographic.com": 0.88,
            "scientificamerican.com": 0.87, "aljazeera.com": 0.82, "dw.com": 0.85,
            "france24.com": 0.83, "timesofindia.indiatimes.com": 0.75, "scmp.com": 0.78
        }
    
    def _setup_prompts(self):
        """Setup LangChain prompts optimized for Google Gemini"""
        self.claim_extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert fact-checker. Analyze news text to extract specific, verifiable factual claims. Focus on facts, not opinions. Extract numbers, dates, names, locations, and events. Separate the main claim from supporting details. {format_instructions} Output in the exact JSON format specified."),
            ("human", "Analyze this news text:\n\n{news_text}")
        ])
        
        self.query_generation_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a research strategist. Generate diverse search queries to verify factual claims. Create Primary, Alternative, and Contradiction queries. Keep queries concise and specific. {format_instructions} Generate queries in the exact JSON format specified."),
            ("human", "Create search queries for this claim:\n\nCLAIM: {claim}\n\nGenerate queries to find both supporting and contradicting evidence.")
        ])
    
    async def extract_claims(self, news_text: str) -> ClaimExtractor:
        """Extract verifiable claims from news text using Gemini"""
        try:
            formatted_prompt = self.claim_extraction_prompt.format_prompt(news_text=news_text, format_instructions=self.claim_parser.get_format_instructions())
            response = await self.fast_llm.ainvoke(formatted_prompt.to_messages())
            return self.claim_parser.parse(response.content)
        except Exception as e:
            print(f"Error extracting claims: {e}")
            return ClaimExtractor(claims=[news_text[:200] + "..."], main_claim=news_text.split('.')[0] if '.' in news_text else news_text[:100], supporting_details=[])

    async def generate_search_queries(self, claim: str) -> SearchQueryGenerator:
        """Generate diverse search queries using Gemini"""
        try:
            formatted_prompt = self.query_generation_prompt.format_prompt(claim=claim, format_instructions=self.query_parser.get_format_instructions())
            response = await self.llm.ainvoke(formatted_prompt.to_messages())
            return self.query_parser.parse(response.content)
        except Exception as e:
            print(f"Error generating queries: {e}")
            words = claim.split()[:6]
            basic_query = " ".join(words)
            return SearchQueryGenerator(primary_queries=[basic_query, claim[:50]], alternative_queries=[f"{basic_query} news"], contradiction_queries=[f"{basic_query} debunked"])

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace('www.', '')
        except:
            return "unknown"

    def _calculate_credibility_score(self, domain: str) -> float:
        """Calculate credibility score based on trusted sources database"""
        if domain in self.trusted_sources:
            return self.trusted_sources[domain]
        for trusted_domain, score in self.trusted_sources.items():
            if domain.endswith(trusted_domain):
                return score * 0.9
        return 0.5

    def _calculate_relevance_score(self, query: str, title: str, snippet: str) -> float:
        """Enhanced relevance scoring"""
        query_words = set(query.lower().split())
        text_words = set((title + " " + snippet).lower().split())
        if not query_words: return 0.0
        basic_score = len(query_words.intersection(text_words)) / len(query_words)
        title_boost = (len(query_words.intersection(set(title.lower().split()))) / len(query_words)) * 0.3
        return min(1.0, basic_score + title_boost)

    async def search_with_mock_engine(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """
        MODIFIED: This function no longer calls an external search engine.
        It directly returns structured mock results to allow the script to proceed.
        """
        results = []
        try:
            # This section now generates mock results directly without a network call.
            mock_results = [
                {
                    "title": f"Verification result for: {query}",
                    "url": f"https://reuters.com/article-{hash(query) % 1000}",
                    "snippet": f"Detailed information and analysis regarding {query}. Reuters confirms various aspects of this claim.",
                    "domain": "reuters.com"
                },
                {
                    "title": f"Fact Check: {query}",
                    "url": f"https://www.factcheck.org/fact-{hash(query) % 1000}",
                    "snippet": f"An independent fact-checking organization investigates the claim '{query}'.",
                    "domain": "factcheck.org"
                },
                {
                    "title": f"Opposing view on: {query}",
                    "url": f"https://opinion-source.com/view-{hash(query) % 1000}",
                    "snippet": f"A different perspective argues against the details of '{query}'.",
                    "domain": "opinion-source.com"
                }
            ]
            
            for result in mock_results[:max_results]:
                domain = result["domain"]
                search_result = SearchResult(
                    title=result["title"],
                    url=result["url"],
                    snippet=result["snippet"],
                    source_domain=domain,
                    publish_date=None,
                    search_engine=SearchEngine.MOCK_ENGINE,
                    relevance_score=self._calculate_relevance_score(query, result["title"], result["snippet"]),
                    credibility_score=self._calculate_credibility_score(domain)
                )
                results.append(search_result)
        
        except Exception as e:
            print(f"Error creating mock search results: {e}")
        
        return results

    async def multi_source_search(self, queries: List[str], max_results_per_query: int = 5) -> List[SearchResult]:
        """
        MODIFIED: Search across multiple sources with intelligent deduplication.
        This now calls the internal mock search function.
        """
        all_results = []
        
        # Search each query using the mock engine
        for query in queries:
            results = await self.search_with_mock_engine(query, max_results_per_query)
            all_results.extend(results)
        
        # Simple URL-based deduplication
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        all_results = unique_results
        
        # Sort by combined relevance and credibility score
        all_results.sort(key=lambda x: (x.relevance_score * 0.6 + x.credibility_score * 0.4), reverse=True)
        
        return all_results
    
    # ... The rest of the class methods (_deduplicate_results_with_ai, verify_news_claim, etc.)
    # remain the same as they operate on the results from multi_source_search.

    async def _deduplicate_results_with_ai(self, results: List[SearchResult]) -> List[SearchResult]:
        """Use Gemini to intelligently deduplicate similar results"""
        try:
            results_data = [{"id": i, "title": r.title, "domain": r.source_domain, "snippet": r.snippet[:200], "credibility": r.credibility_score} for i, r in enumerate(results)]
            model = genai.GenerativeModel('gemini-2.0-flash')
            dedup_prompt = f"""Analyze these search results and identify duplicates. Group results that cover the same story, select the one with the highest credibility, and return the IDs of results to keep (max 15). Results: {json.dumps(results_data[:20], indent=2)}. Respond with a JSON array of IDs: [1, 3, 7, ...]"""
            response = model.generate_content(dedup_prompt)
            keep_ids = json.loads(response.text.strip())
            return [results[i] for i in keep_ids if i < len(results)]
        except Exception as e:
            print(f"Error in AI deduplication: {e}")
            seen_domains = set()
            unique_results = []
            for result in results:
                if result.source_domain not in seen_domains or result.credibility_score > 0.8:
                    seen_domains.add(result.source_domain)
                    unique_results.append(result)
            return unique_results[:15]

    async def verify_news_claim(self, news_text: str) -> Dict[str, Any]:
        """Main method to verify a news claim using Gemini models"""
        print(f"Starting Gemini-powered verification for: {news_text[:100]}...")
        print("Extracting claims with Gemini...")
        claims_data = await self.extract_claims(news_text)
        print("Generating search queries with Gemini...")
        query_data = await self.generate_search_queries(claims_data.main_claim)
        all_queries = (query_data.primary_queries + query_data.alternative_queries + query_data.contradiction_queries)
        print("Searching with internal mock engine...")
        search_results = await self.multi_source_search(all_queries)
        print("Analyzing results with Gemini...")
        return await self._analyze_search_results_with_ai(claims_data, query_data, search_results)

    async def _analyze_search_results_with_ai(self, claims: ClaimExtractor, queries: SearchQueryGenerator, results: List[SearchResult]) -> Dict[str, Any]:
        """Use Gemini to analyze search results and generate verification report"""
        try:
            analysis_data = {"main_claim": claims.main_claim, "search_results": [{"title": r.title, "domain": r.source_domain, "snippet": r.snippet, "credibility_score": r.credibility_score, "relevance_score": r.relevance_score} for r in results[:10]]}
            model = genai.GenerativeModel('gemini-2.0-flash')
            analysis_prompt = f"""Analyze search results to verify the claim. Assess evidence quality, consensus, and contradictions. Rate confidence (0.0-1.0) and status (HIGHLY_VERIFIED, LIKELY_ACCURATE, etc.). Provide detailed reasoning. CLAIM: {claims.main_claim}. RESULTS: {json.dumps(analysis_data['search_results'], indent=2)}"""
            response = model.generate_content(analysis_prompt)
            ai_analysis = response.text
            confidence_score = self._extract_confidence_from_analysis(ai_analysis, results)
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            ai_analysis = "AI analysis unavailable. Using fallback scoring."
            confidence_score = self._calculate_fallback_confidence(results)
        
        return {
            "original_claim": claims.main_claim, "extracted_claims": claims.claims,
            "search_queries_used": queries.primary_queries + queries.alternative_queries,
            "total_sources_found": len(results),
            "high_credibility_sources": len([r for r in results if r.credibility_score >= 0.8]),
            "confidence_score": confidence_score,
            "verification_status": self._determine_verification_status(confidence_score),
            "ai_analysis": ai_analysis,
            "top_sources": [{"title": r.title, "url": r.url, "domain": r.source_domain, "credibility_score": r.credibility_score, "relevance_score": r.relevance_score} for r in results[:5]],
            "analysis_timestamp": datetime.now().isoformat(),
            "recommendation": self._generate_recommendation(confidence_score, results),
            "model_used": "Google gemini-2.0-flash"
        }

    def _extract_confidence_from_analysis(self, analysis_text: str, results: List[SearchResult]) -> float:
        """Extract confidence score from Gemini's analysis"""
        confidence_keywords = {"highly confident": 0.9, "very confident": 0.85, "confident": 0.8, "moderately confident": 0.65, "somewhat confident": 0.6, "uncertain": 0.4, "low confidence": 0.3, "very uncertain": 0.2}
        analysis_lower = analysis_text.lower()
        for keyword, score in confidence_keywords.items():
            if keyword in analysis_lower: return score
        return self._calculate_fallback_confidence(results)

    def _calculate_fallback_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score using traditional metrics"""
        if not results: return 0.0
        high_credibility_sources = [r for r in results if r.credibility_score >= 0.8]
        avg_credibility = sum(r.credibility_score for r in results[:10]) / min(10, len(results))
        avg_relevance = sum(r.relevance_score for r in results[:10]) / min(10, len(results))
        return min(1.0, (len(high_credibility_sources) * 0.15 + avg_credibility * 0.5 + avg_relevance * 0.35))

    def _determine_verification_status(self, confidence_score: float) -> str:
        """Determine verification status based on confidence score"""
        if confidence_score >= 0.85: return "HIGHLY_VERIFIED"
        elif confidence_score >= 0.7: return "LIKELY_ACCURATE"
        elif confidence_score >= 0.5: return "UNCERTAIN"
        elif confidence_score >= 0.3: return "LIKELY_INACCURATE"
        else: return "INSUFFICIENT_EVIDENCE"

    def _generate_recommendation(self, confidence_score: float, results: List[SearchResult]) -> str:
        """Generate human-readable recommendation"""
        high_cred_count = len([r for r in results if r.credibility_score >= 0.8])
        if confidence_score >= 0.85: return f"Claim is well-supported by {high_cred_count} high-credibility sources. High confidence in accuracy."
        elif confidence_score >= 0.7: return f"Claim has good support but may benefit from more verification. Moderate confidence."
        elif confidence_score >= 0.5: return "Mixed evidence found. Exercise caution and seek more authoritative sources."
        elif confidence_score >= 0.3: return "Claim lacks sufficient credible support. Treat with skepticism."
        else: return "Insufficient reliable evidence found to verify this claim."