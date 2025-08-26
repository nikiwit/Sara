#!/usr/bin/env python3
"""
Test script for language detection and ambiguity handling improvements.
Run this to verify the implementations work correctly.
"""

import sys
import os

# Add SARA directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from query_handling.language_handler import LanguageHandler
from query_handling.ambiguity_handler import AmbiguityHandler

def test_language_detection():
    """Test language detection functionality."""
    print("=== LANGUAGE DETECTION TESTS ===")
    handler = LanguageHandler()
    
    test_cases = [
        ("Hello, how are you?", "en", False),
        ("¿Cómo puedo renovar mi visa?", "es", True),
        ("你好吗？", "zh", True),
        ("как дела?", "ru", True),
        ("How do I renew my passport?", "pl", False),  # With confidence thresholding, low-confidence detections should pass through
        ("Bonjour comment allez-vous?", "fr", True),  # French should be blocked
        ("Hallo wie geht es dir?", "de", True),  # German should be blocked
    ]
    
    for query, expected_lang, should_block in test_cases:
        detected_lang = handler.detect_language(query)
        is_blocked, response = handler.handle_query(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Detected: {detected_lang} | Expected: {expected_lang}")
        print(f"Blocked: {is_blocked} | Expected: {should_block}")
        
        if is_blocked:
            print(f"Response: {response[:100]}...")
        
        # Verify results
        status = "✅ PASS" if (is_blocked == should_block) else "❌ FAIL"
        print(f"Status: {status}")

def test_ambiguity_detection():
    """Test ambiguous query detection and clarification."""
    print("\n\n=== AMBIGUITY DETECTION TESTS ===")
    handler = AmbiguityHandler()
    
    test_cases = [
        ("How do I renew it?", True),
        ("What are the requirements?", True),
        ("How do I renew my visa?", False),
        ("What are the visa renewal requirements?", False),  # Has clear context - should NOT be ambiguous
        ("Where do I submit?", True),
        ("Where do I submit my application?", False),
    ]
    
    for query, should_be_ambiguous in test_cases:
        is_ambiguous = handler.is_ambiguous(query)
        ambiguity_score = handler.calculate_ambiguity_score(query)
        
        print(f"\nQuery: '{query}'")
        print(f"Ambiguous: {is_ambiguous} | Expected: {should_be_ambiguous} | Score: {ambiguity_score:.2f}")
        
        if is_ambiguous:
            clarification = handler.get_clarification(query)
            print(f"Clarification: {clarification[:100]}...")
        
        # Verify results
        status = "✅ PASS" if (is_ambiguous == should_be_ambiguous) else f"❌ FAIL (Score: {ambiguity_score:.2f} vs threshold: {handler.confidence_threshold})"
        print(f"Status: {status}")

def test_integration():
    """Test integration of both handlers."""
    print("\n\n=== INTEGRATION TESTS ===")
    
    # This simulates what happens in the conversation handler
    language_handler = LanguageHandler()
    ambiguity_handler = AmbiguityHandler()
    
    test_queries = [
        "¿Cómo puedo renovar mi visa?",  # Non-English
        "How do I renew it?",           # Ambiguous
        "How do I renew my student visa?",  # Clear and valid
        "What are the requirements?",    # Ambiguous
    ]
    
    for query in test_queries:
        print(f"\n--- Processing: '{query}' ---")
        
        # Step 1: Language check
        should_block, lang_response = language_handler.handle_query(query)
        if should_block:
            print(f"🚫 BLOCKED (Language): {lang_response[:80]}...")
            continue
            
        # Step 2: Ambiguity check  
        if ambiguity_handler.is_ambiguous(query):
            clarification = ambiguity_handler.get_clarification(query)
            print(f"❓ CLARIFICATION NEEDED: {clarification[:80]}...")
            continue
            
        # Step 3: Normal processing
        print("✅ PROCEEDING TO RAG PIPELINE")

if __name__ == "__main__":
    print("Testing SARA Language Detection and Ambiguity Handling")
    print("=" * 60)
    
    try:
        test_language_detection()
        test_ambiguity_detection() 
        test_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED!")
        print("If you see failures, check the implementation logic.")
        print("These handlers are now integrated into your SARA chatbot.")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure to install langdetect: pip install langdetect==1.0.9")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check the implementation for issues.")