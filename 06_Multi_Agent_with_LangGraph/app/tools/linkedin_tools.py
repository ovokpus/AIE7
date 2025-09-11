from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool
import re


@tool
def create_linkedin_post(
    content: Annotated[str, "The main content for the LinkedIn post"],
    paper_title: Annotated[str, "Title of the ML paper"],
    key_insights: Annotated[List[str], "List of key insights from the paper"],
    target_audience: Annotated[str, "Target audience (professional, academic, general)"] = "professional",
    include_technical_details: Annotated[bool, "Whether to include technical details"] = True,
    max_hashtags: Annotated[int, "Maximum number of hashtags"] = 10,
    tone: Annotated[str, "Tone of the post (professional, casual, academic)"] = "professional"
) -> Annotated[str, "Generated LinkedIn post"]:
    """Create a LinkedIn post about a machine learning paper."""
    
    # Choose appropriate emoji based on tone
    emoji_map = {
        "professional": "🚀",
        "academic": "📚", 
        "casual": "💡"
    }
    opening_emoji = emoji_map.get(tone, "🚀")
    
    # Create engaging opening based on audience
    if target_audience == "academic":
        opening = f"{opening_emoji} **New Research Alert: {paper_title}**"
    elif target_audience == "general":
        opening = f"{opening_emoji} **Exciting breakthrough in AI!**"
    else:  # professional
        opening = f"{opening_emoji} **Transforming the Future of AI: {paper_title}**"
    
    # Format the main content
    post = f"{opening}\n\n{content}\n\n"
    
    # Add key insights section
    if key_insights:
        post += "💡 **Key Takeaways:**\n"
        for i, insight in enumerate(key_insights[:5], 1):  # Limit to 5 insights
            post += f"\n{i}. {insight}"
        post += "\n\n"
    
    # Add engagement question based on audience
    if target_audience == "academic":
        post += "What are your thoughts on this methodology? How do you see it advancing the field?\n\n"
    elif target_audience == "general":
        post += "What excites you most about AI developments like this?\n\n"
    else:  # professional
        post += "What are your thoughts on this research? How do you see it impacting your industry?\n\n"
    
    # Generate relevant hashtags
    hashtags = generate_linkedin_hashtags(paper_title, key_insights, max_hashtags)
    post += " ".join(hashtags)
    
    return post


@tool
def verify_technical_accuracy(
    post_content: Annotated[str, "LinkedIn post content to verify"],
    paper_reference: Annotated[str, "Reference information about the paper"],
) -> Annotated[str, "Technical accuracy assessment"]:
    """Verify the technical accuracy of claims made in the LinkedIn post."""
    
    # Extract technical claims from the post
    technical_terms = extract_technical_terms(post_content)
    
    # Check for common accuracy issues
    accuracy_issues = []
    recommendations = []
    
    # Check for overstated claims
    overstated_patterns = [
        r"revolutionary", r"breakthrough", r"unprecedented", 
        r"solves all", r"perfect", r"100%", r"completely"
    ]
    
    for pattern in overstated_patterns:
        if re.search(pattern, post_content, re.IGNORECASE):
            accuracy_issues.append(f"Potentially overstated claim detected: '{pattern}'")
            recommendations.append("Consider using more measured language")
    
    # Check for proper attribution
    if "et al" not in post_content and "by" not in post_content.lower():
        accuracy_issues.append("Missing author attribution")
        recommendations.append("Add proper attribution to paper authors")
    
    # Generate verification report
    accuracy_score = max(0.0, 1.0 - (len(accuracy_issues) * 0.2))
    
    verification_report = f"""
TECHNICAL VERIFICATION REPORT:
=============================

POST CONTENT ANALYZED:
{post_content[:500]}...

TECHNICAL TERMS IDENTIFIED:
{', '.join(technical_terms) if technical_terms else 'None detected'}

ACCURACY ASSESSMENT:
Score: {accuracy_score:.2f}/1.0

ISSUES IDENTIFIED:
{chr(10).join(f'- {issue}' for issue in accuracy_issues) if accuracy_issues else '- No major issues detected'}

RECOMMENDATIONS:
{chr(10).join(f'- {rec}' for rec in recommendations) if recommendations else '- Post appears technically sound'}

STATUS: {'APPROVED' if accuracy_score >= 0.7 else 'NEEDS REVISION'}
"""
    return verification_report


@tool
def check_linkedin_style(
    post_content: Annotated[str, "LinkedIn post content to check"],
) -> Annotated[str, "Style compliance assessment"]:
    """Check if the post follows LinkedIn best practices and professional tone."""
    
    style_issues = []
    recommendations = []
    
    # Check post length
    char_count = len(post_content)
    if char_count > 3000:
        style_issues.append(f"Post too long ({char_count} chars, limit: 3000)")
        recommendations.append("Shorten content for better engagement")
    elif char_count < 100:
        style_issues.append(f"Post too short ({char_count} chars)")
        recommendations.append("Add more valuable content")
    
    # Check for appropriate emoji usage
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', post_content))
    if emoji_count > 10:
        style_issues.append(f"Too many emojis ({emoji_count})")
        recommendations.append("Reduce emoji usage for professional tone")
    elif emoji_count == 0:
        style_issues.append("No emojis used")
        recommendations.append("Add 1-3 relevant emojis for engagement")
    
    # Check for hashtags
    hashtag_count = len(re.findall(r'#\w+', post_content))
    if hashtag_count > 20:
        style_issues.append(f"Too many hashtags ({hashtag_count})")
        recommendations.append("Limit hashtags to 5-10 for better reach")
    elif hashtag_count == 0:
        style_issues.append("No hashtags found")
        recommendations.append("Add relevant hashtags for discoverability")
    
    # Check for engagement elements
    has_question = '?' in post_content
    if not has_question:
        style_issues.append("Missing engagement question")
        recommendations.append("Add a question to encourage comments")
    
    # Check formatting
    has_line_breaks = '\n' in post_content
    if not has_line_breaks:
        style_issues.append("Poor formatting - no line breaks")
        recommendations.append("Add line breaks for better readability")
    
    # Calculate style score
    style_score = max(0.0, 1.0 - (len(style_issues) * 0.15))
    
    style_report = f"""
LINKEDIN STYLE ASSESSMENT:
=========================

POST ANALYZED:
{post_content[:300]}...

METRICS:
- Character count: {char_count}
- Emoji count: {emoji_count}
- Hashtag count: {hashtag_count}
- Has engagement question: {'Yes' if has_question else 'No'}
- Proper formatting: {'Yes' if has_line_breaks else 'No'}

STYLE SCORE: {style_score:.2f}/1.0

ISSUES IDENTIFIED:
{chr(10).join(f'- {issue}' for issue in style_issues) if style_issues else '- No major style issues'}

RECOMMENDATIONS:
{chr(10).join(f'- {rec}' for rec in recommendations) if recommendations else '- Post follows LinkedIn best practices'}

STATUS: {'LINKEDIN READY' if style_score >= 0.7 else 'NEEDS STYLE IMPROVEMENTS'}
"""
    return style_report


def extract_technical_terms(text: str) -> List[str]:
    """Extract technical terms from text."""
    # Common ML/AI technical terms
    technical_patterns = [
        r'\b(?:neural network|transformer|attention|BERT|GPT|CNN|RNN|LSTM)\b',
        r'\b(?:machine learning|deep learning|artificial intelligence|AI|ML|DL)\b',
        r'\b(?:algorithm|model|dataset|training|inference|optimization)\b',
        r'\b(?:accuracy|precision|recall|F1|loss|gradient|backpropagation)\b',
        r'\b(?:supervised|unsupervised|reinforcement|learning)\b'
    ]
    
    technical_terms = []
    for pattern in technical_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        technical_terms.extend(matches)
    
    return list(set(technical_terms))  # Remove duplicates


def generate_linkedin_hashtags(paper_title: str, key_insights: List[str], max_hashtags: int = 10) -> List[str]:
    """Generate relevant hashtags for a LinkedIn post about an ML paper."""
    
    # Base hashtags for ML content
    base_hashtags = ["#MachineLearning", "#AI", "#Research", "#Innovation", "#TechTrends"]
    
    # Extract topic-specific hashtags from paper title and insights
    text = f"{paper_title} {' '.join(key_insights)}"
    
    topic_hashtags = []
    
    # Common ML topics and their hashtags
    topic_mapping = {
        r'natural language|nlp|text|language': '#NLP',
        r'computer vision|cv|image|visual': '#ComputerVision',
        r'transformer|attention|bert|gpt': '#Transformers',
        r'deep learning|neural network': '#DeepLearning',
        r'reinforcement learning|rl': '#ReinforcementLearning',
        r'data science|analytics': '#DataScience',
        r'python|pytorch|tensorflow': '#Python',
        r'automation|efficiency': '#Automation',
        r'business|industry|enterprise': '#BusinessAI',
        r'algorithm|optimization': '#Algorithms'
    }
    
    for pattern, hashtag in topic_mapping.items():
        if re.search(pattern, text, re.IGNORECASE):
            topic_hashtags.append(hashtag)
    
    # Combine and limit hashtags
    all_hashtags = base_hashtags + topic_hashtags
    return all_hashtags[:max_hashtags] 