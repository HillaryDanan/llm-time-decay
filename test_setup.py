#!/usr/bin/env python3
"""
Test script to verify installation and API keys.

Run: python3 test_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required packages are installed."""
    print("Testing imports...")
    try:
        import numpy
        print("✓ numpy")
    except ImportError:
        print("✗ numpy - run: pip install numpy")
        return False
    
    try:
        import scipy
        print("✓ scipy")
    except ImportError:
        print("✗ scipy - run: pip install scipy")
        return False
    
    try:
        import openai
        print("✓ openai")
    except ImportError:
        print("✗ openai - run: pip install openai")
        return False
    
    try:
        import anthropic
        print("✓ anthropic")
    except ImportError:
        print("✗ anthropic - run: pip install anthropic")
        return False
    
    try:
        import matplotlib
        print("✓ matplotlib")
    except ImportError:
        print("✗ matplotlib - run: pip install matplotlib")
        return False
    
    return True

def test_config():
    """Test configuration and API keys."""
    print("\nTesting configuration...")
    try:
        from config import validate_config, API_KEYS
        
        if validate_config():
            print("✓ Configuration valid")
            print(f"  OpenAI key: {'✓' if API_KEYS.get('openai') else '✗ Missing'}")
            print(f"  Anthropic key: {'✓' if API_KEYS.get('anthropic') else '✗ Missing'}")
            print(f"  Gemini key: {'✓' if API_KEYS.get('gemini') else '✗ Missing'}")
            return True
        else:
            print("✗ Configuration invalid - check .env file")
            return False
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False

def test_generator():
    """Test prompt generation."""
    print("\nTesting prompt generator...")
    try:
        from generator import PromptGenerator
        
        gen = PromptGenerator()
        
        # Test fractional depth
        prompt = gen.generate(2.5)
        if "T0" in prompt and "T2" in prompt:
            print("✓ Fractional depth generation works")
        else:
            print("✗ Problem with fractional depth generation")
            return False
        
        # Test integer depth
        prompt = gen.generate(3.0)
        if "thinking about thinking about thinking" in prompt:
            print("✓ Integer depth generation works")
        else:
            print("✗ Problem with integer depth generation")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error in generator: {e}")
        return False

def test_scorer():
    """Test scoring system."""
    print("\nTesting scorer...")
    try:
        from scorer import TemporalScorer
        
        scorer = TemporalScorer()
        
        # Test with sample response
        response = """At time T0, I am thinking. 
        At time T1, I realize that I was thinking at T0, 
        so now I am thinking about thinking."""
        
        result = scorer.score(response, 2.0)
        
        if 0 <= result.total_score <= 1:
            print(f"✓ Scoring works (sample score: {result.total_score:.3f})")
        else:
            print("✗ Scoring out of expected range")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error in scorer: {e}")
        return False

def test_api_connection():
    """Quick test of API connections (optional, costs tokens)."""
    print("\nDo you want to test API connections? (y/n)")
    response = input().lower()
    
    if response != 'y':
        print("Skipping API tests")
        return True
    
    print("\nTesting API connections...")
    
    try:
        from config import API_KEYS
        
        # Test OpenAI
        if API_KEYS.get('openai'):
            from openai import OpenAI
            client = OpenAI(api_key=API_KEYS['openai'])
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10
            )
            print("✓ OpenAI API working")
        else:
            print("⚠️  OpenAI API key missing")
        
        # Test Anthropic
        if API_KEYS.get('anthropic'):
            from anthropic import Anthropic
            client = Anthropic(api_key=API_KEYS['anthropic'])
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10
            )
            print("✓ Anthropic API working")
        else:
            print("⚠️  Anthropic API key missing")
        
        # Test Gemini
        if API_KEYS.get('gemini'):
            import google.generativeai as genai
            genai.configure(api_key=API_KEYS['gemini'])
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            response = model.generate_content("Say 'test successful'")
            print("✓ Gemini API working")
        else:
            print("⚠️  Gemini API key missing")
        
        return True
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 50)
    print("LLM TIME DECAY - SETUP VERIFICATION")
    print("=" * 50)
    
    all_good = True
    
    # Test imports
    if not test_imports():
        all_good = False
    
    # Test config
    if not test_config():
        all_good = False
    
    # Test generator
    if not test_generator():
        all_good = False
    
    # Test scorer
    if not test_scorer():
        all_good = False
    
    # Optional API test
    test_api_connection()
    
    print("\n" + "=" * 50)
    if all_good:
        print("✅ ALL TESTS PASSED - Ready to run experiments!")
        print("\nRun experiment with:")
        print("  python3 src/runner.py --models all --depths fractional --trials 50")
    else:
        print("⚠️  Some tests failed - fix issues above before running")
    print("=" * 50)

if __name__ == "__main__":
    main()