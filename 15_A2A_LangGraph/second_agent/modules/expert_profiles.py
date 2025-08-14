"""
Expert Profile Definitions for Goal-Oriented Agent Behavior.

This module defines the ExpertProfile class and a collection of specialized
expert agents that demonstrate goal-oriented behavior in A2A communications.
Each expert has specific research missions, quality standards, and follow-up
strategies that drive persistent, high-quality interactions.

The module includes 4 predefined expert profiles covering different domains:
    - Machine Learning Research (Dr. Sarah Chen)
    - AI Architecture Research (Prof. Marcus Rodriguez)  
    - Business/Startup Applications (Alex Kim)
    - AI Security and Safety (Dr. Emma Watson)

Key Features:
    - Goal-oriented expert behavior with specific missions
    - Quality evaluation using domain-specific criteria
    - Automatic follow-up question generation
    - Persistent expert personality across conversations
    - Comprehensive persona context generation

Example:
    >>> profile = EXPERT_PROFILES['ml_expert_kimi']
    >>> context = profile.get_persona_context()
    >>> quality_score = profile.evaluate_response_quality(response)
"""

from typing import List, Dict


class ExpertProfile:
    """Goal-oriented expert agent with specific research missions and quality standards.
    
    This class represents a specialized expert agent that maintains persistent
    behavior across conversations. Each expert has specific goals, quality 
    standards, and domain expertise that drives their interaction patterns
    and evaluation criteria.
    
    The expert profiles demonstrate sophisticated agent behavior including:
        - Persistent research missions and goals
        - Quality evaluation based on domain expertise
        - Automatic follow-up question generation
        - Learning and adaptation over time
        
    Attributes:
        name (str): The expert's professional name/identity
        identity (str): Detailed description of the expert's role and background
        current_goal (str): Specific research mission or objective
        quality_standards (List[str]): Criteria for evaluating response quality
        follow_up_strategy (str): Approach for generating follow-up questions
        domain_expertise (List[str]): Areas of specialized knowledge
        questions_asked (int): Counter for tracking conversation depth
        satisfaction_level (int): Current satisfaction with responses (0-10 scale)
        
    Example:
        >>> expert = ExpertProfile(
        ...     name="Dr. Sarah Chen",
        ...     identity="an expert in Machine Learning",
        ...     current_goal="learn about what makes Kimi K2 so incredible",
        ...     quality_standards=["not satisfied with surface level answers"],
        ...     follow_up_strategy="Ask for technical details if initial answer lacks depth",
        ...     domain_expertise=["machine learning", "neural networks"]
        ... )
        >>> context = expert.get_persona_context()
        >>> score = expert.evaluate_response_quality("Brief response about ML")
    """
    
    def __init__(self, 
                 name: str,
                 identity: str, 
                 current_goal: str, 
                 quality_standards: List[str],
                 follow_up_strategy: str,
                 domain_expertise: List[str]) -> None:
        """Initialize an expert profile with goals and standards.
        
        Args:
            name (str): Professional name/identity of the expert
            identity (str): Detailed role description (e.g., "an expert in Machine Learning")
            current_goal (str): Specific research mission or objective
            quality_standards (List[str]): List of quality criteria for evaluating responses
            follow_up_strategy (str): Strategy for generating follow-up questions
            domain_expertise (List[str]): Areas of specialized knowledge and expertise
        """
        self.name = name
        self.identity = identity
        self.current_goal = current_goal
        self.quality_standards = quality_standards
        self.follow_up_strategy = follow_up_strategy
        self.domain_expertise = domain_expertise
        self.questions_asked = 0
        self.satisfaction_level = 0  # 0-10 scale
        
    def get_persona_context(self) -> str:
        """Generate comprehensive persona context for A2A communications.
        
        Creates a detailed persona description that will be included in A2A
        calls to provide context about the expert's goals, standards, and
        approach. This context helps the receiving agent tailor its response
        to meet the expert's specific requirements.
        
        Returns:
            str: Formatted persona context string containing identity, goals,
                quality standards, domain expertise, and follow-up strategy
                
        Example:
            >>> expert = ExpertProfile(...)
            >>> context = expert.get_persona_context()
            >>> # Context includes: "You are an expert in Machine Learning..."
        """
        standards_text = " and ".join(self.quality_standards)
        return f"""You are {self.identity}. Your current goal is: {self.current_goal}.
        
                    Quality Standards: {standards_text}
                            
                    Domain Expertise: {', '.join(self.domain_expertise)}

                    Follow-up Strategy: {self.follow_up_strategy}

                    Remember: You are a persistent expert who won't settle for superficial answers. Ask follow-up questions if needed."""

    def evaluate_response_quality(self, response: str) -> int:
        """Evaluate response quality against expert's standards and domain expertise.
        
        Analyzes the response content to determine if it meets the expert's
        quality standards and contains sufficient depth for their research goals.
        Uses multiple criteria including length, domain relevance, source 
        citations, and alignment with specific quality standards.
        
        Args:
            response (str): The response content to evaluate
            
        Returns:
            int: Quality score from 0-10 where:
                - 0-3: Poor quality, definitely needs follow-up
                - 4-6: Moderate quality, may need follow-up
                - 7-10: High quality, likely satisfactory
                
        Evaluation Criteria:
            - Response length and depth
            - Presence of research/study references
            - Source citations and links
            - Domain expertise keyword matching
            - Alignment with specific quality standards
            
        Example:
            >>> expert = ExpertProfile(...)
            >>> score = expert.evaluate_response_quality("Brief ML overview")
            >>> # Returns lower score due to lack of depth
            >>> score = expert.evaluate_response_quality("Detailed ML paper with sources...")
            >>> # Returns higher score due to depth and sources
        """
        score = 5  # baseline
        
        # Check for depth indicators
        if len(response) > 500:
            score += 1
        if "research" in response.lower() or "study" in response.lower():
            score += 1
        if "source" in response.lower() or "paper" in response.lower():
            score += 1
        if any(expertise.lower() in response.lower() for expertise in self.domain_expertise):
            score += 1
            
        # Check against quality standards
        if "not satisfied with surface level" in " ".join(self.quality_standards):
            if len(response.split()) < 100:  # Too short
                score -= 2
            if "detailed" in response.lower() or "technical" in response.lower():
                score += 1
                
        if "want sources" in " ".join(self.quality_standards):
            if "http" in response or "arxiv" in response.lower() or "doi" in response.lower():
                score += 2
            else:
                score -= 1
                
        return min(10, max(0, score))


# Pre-defined expert profiles for different scenarios
EXPERT_PROFILES = {
    "ml_expert_kimi": ExpertProfile(
        name="Dr. Sarah Chen",
        identity="an expert in Machine Learning",
        current_goal="learn about what makes Kimi K2 so incredible",
        quality_standards=[
            "not satisfied with surface level answers",
            "want sources to read to verify information"
        ],
        follow_up_strategy="If initial answer lacks depth, ask for technical details, papers, or implementation specifics",
        domain_expertise=["machine learning", "neural networks", "language models", "AI architectures"]
    ),
    
    "ai_researcher_transformers": ExpertProfile(
        name="Prof. Marcus Rodriguez",
        identity="an AI researcher specializing in transformer architectures",
        current_goal="understand the latest innovations in attention mechanisms and their practical applications",
        quality_standards=[
            "need academic rigor and citations",
            "require technical implementation details"
        ],
        follow_up_strategy="Demand mathematical explanations and code examples when concepts are mentioned",
        domain_expertise=["transformers", "attention mechanisms", "deep learning", "NLP"]
    ),
    
    "startup_founder_ai": ExpertProfile(
        name="Alex Kim",
        identity="a startup founder building an AI-powered product",
        current_goal="evaluate which AI technologies to build on and understand their business implications",
        quality_standards=[
            "need practical implementation details",
            "require cost analysis and ROI data"
        ],
        follow_up_strategy="Ask for real-world examples, pricing, and scalability concerns",
        domain_expertise=["AI APIs", "business strategy", "product development", "scaling AI"]
    ),
    
    "security_expert_ai": ExpertProfile(
        name="Dr. Emma Watson",
        identity="a cybersecurity expert investigating AI system vulnerabilities",
        current_goal="understand security risks in large language models and mitigation strategies",
        quality_standards=[
            "need concrete examples of vulnerabilities",
            "require mitigation strategies with evidence"
        ],
        follow_up_strategy="Ask for specific attack vectors and defense mechanisms with technical proof",
        domain_expertise=["AI security", "prompt injection", "model safety", "adversarial attacks"]
    )
}


def get_expert_profile(expert_id: str) -> ExpertProfile:
    """Get expert profile by ID"""
    return EXPERT_PROFILES.get(expert_id, EXPERT_PROFILES["ml_expert_kimi"])


def list_available_experts() -> Dict[str, str]:
    """List all available expert profiles"""
    return {
        expert_id: profile.name
        for expert_id, profile in EXPERT_PROFILES.items()
    }
