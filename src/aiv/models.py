"""SQLAlchemy database models"""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Prompt(Base):
    """Buyer-intent prompts for B2B SaaS categories"""
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(100), nullable=False, index=True)
    text = Column(Text, nullable=False)
    intent = Column(String(50), nullable=False)  # best-for, alternative, comparison, pricing, use-case
    created_at = Column(DateTime, default=datetime.utcnow)
    
    responses = relationship("Response", back_populates="prompt")
    search_results = relationship("SearchResult", back_populates="prompt")
    
    __table_args__ = (
        Index("ix_prompts_category_intent", "category", "intent"),
    )


class Response(Base):
    """LLM responses with grounding metadata"""
    __tablename__ = "responses"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False, index=True)
    model = Column(String(50), nullable=False)  # e.g., "gemini-1.5-pro"
    text = Column(Text, nullable=False)
    raw_json = Column(Text, nullable=False)  # Full API response for debugging
    created_at = Column(DateTime, default=datetime.utcnow)
    
    prompt = relationship("Prompt", back_populates="responses")
    mentions = relationship("Mention", back_populates="response")
    citations = relationship("Citation", back_populates="response")


class Mention(Base):
    """Brand mentions extracted from LLM responses"""
    __tablename__ = "mentions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    response_id = Column(Integer, ForeignKey("responses.id"), nullable=False, index=True)
    brand = Column(String(100), nullable=False)
    rank = Column(Integer, nullable=False)  # Order of appearance (1 = first)
    sentiment = Column(String(20), nullable=False)  # positive, neutral, negative
    
    response = relationship("Response", back_populates="mentions")
    
    __table_args__ = (
        Index("ix_mentions_response_brand", "response_id", "brand"),
    )


class Citation(Base):
    """Citations (grounding URLs) from LLM responses"""
    __tablename__ = "citations"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    response_id = Column(Integer, ForeignKey("responses.id"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    domain = Column(String(255), nullable=False, index=True)
    position = Column(Integer, nullable=False)  # Position in grounding sources
    
    response = relationship("Response", back_populates="citations")
    
    __table_args__ = (
        Index("ix_citations_response_domain", "response_id", "domain"),
    )


class SearchResult(Base):
    """Search results from Brave Search API"""
    __tablename__ = "search_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(Integer, ForeignKey("prompts.id"), nullable=False, index=True)
    url = Column(Text, nullable=False)
    domain = Column(String(255), nullable=False, index=True)
    search_rank = Column(Integer, nullable=False)  # 1-10
    
    prompt = relationship("Prompt", back_populates="search_results")
    
    __table_args__ = (
        UniqueConstraint("prompt_id", "url", name="uq_search_results_prompt_url"),
        Index("ix_search_results_prompt_rank", "prompt_id", "search_rank"),
    )


class Page(Base):
    """Scraped web page content"""
    __tablename__ = "pages"
    
    url = Column(Text, primary_key=True)
    status = Column(Integer, nullable=False)  # HTTP status code
    title = Column(Text)
    text = Column(Text)
    html = Column(Text)  # Raw HTML for feature extraction
    fetched_at = Column(DateTime, default=datetime.utcnow)
    
    features = relationship("PageFeature", back_populates="page", uselist=False)


class PageFeature(Base):
    """Computed features for each scraped page"""
    __tablename__ = "page_features"
    
    url = Column(Text, ForeignKey("pages.url"), primary_key=True)
    word_count = Column(Integer, default=0)
    h2_count = Column(Integer, default=0)
    h3_count = Column(Integer, default=0)
    list_count = Column(Integer, default=0)
    has_table = Column(Integer, default=0)  # 0/1
    has_faq = Column(Integer, default=0)  # 0/1
    has_json_ld = Column(Integer, default=0)  # 0/1
    json_ld_types = Column(Text)  # Comma-separated schema.org types
    days_since_published = Column(Integer)
    days_since_updated = Column(Integer)
    domain_type = Column(String(50))  # reddit, review_site, vendor, blog, news, other
    category_brand_mentions = Column(Integer, default=0)
    prompt_similarity = Column(Integer)  # Scaled cosine similarity * 10000
    
    page = relationship("Page", back_populates="features")