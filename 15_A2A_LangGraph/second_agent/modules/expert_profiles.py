# second_agent/modules/expert_profiles.py
"""
Expert Profile definitions for goal-oriented agent behavior
"""
from typing import List, Dict


class ExpertProfile:
    """Represents a specific expert agent with goals and standards"""
    def __init__(self, 
                 name: str,
                 identity: str, 
                 current_goal: str, 
                 quality_standards: List[str],
                 follow_up_strategy: str,
                 domain_expertise: List[str]):
        self.name = name
        self.identity = identity
        self.current_goal = current_goal
        self.quality_standards = quality_standards
        self.follow_up_strategy = follow_up_strategy
        self.domain_expertise = domain_expertise
        self.questions_asked = 0
        self.satisfaction_level = 0  # 0-10 scale
        
    def get_persona_context(self) -> str:
        """Generate persona context for A2A calls"""
        standards_text = " and ".join(self.quality_standards)
        return f"""You are {self.identity}. Your current goal is: {self.current_goal}.
        
                    Quality Standards: {standards_text}
                            
                    Domain Expertise: {', '.join(self.domain_expertise)}

                    Follow-up Strategy: {self.follow_up_strategy}

                    Remember: You are a persistent expert who won't settle for superficial answers. Ask follow-up questions if needed."""

    def evaluate_response_quality(self, response: str) -> int:
        """Evaluate if response meets quality standards (0-10)"""
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
