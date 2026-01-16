#!/usr/bin/env python3
"""
Test script for the User Study Backend API.
Run this after starting the server to verify everything works.

Usage:
    python test_api.py              # Run all tests with real API calls
    python test_api.py --mock       # Run tests (expects mock mode in app.py)
"""
import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_success(text):
    print(f"  ✓ {text}")


def print_error(text):
    print(f"  ✗ {text}")


def print_info(text):
    print(f"    {text}")


def test_health():
    """Test health endpoint."""
    print_header("Test 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        data = response.json()
        
        print_info(f"Status: {response.status_code}")
        print_info(f"botorch_available: {data.get('botorch_available')}")
        print_info(f"num_models: {data.get('num_models')}")
        print_info(f"openrouter_configured: {data.get('openrouter_configured')}")
        
        if response.status_code == 200:
            print_success("Health check passed")
            return True
        else:
            print_error("Health check failed")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Could not connect to server")
        print_info("Make sure the server is running: uvicorn app:app --reload")
        return False


def test_models():
    """Test models listing endpoint."""
    print_header("Test 2: List Models")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        data = response.json()
        
        print_info(f"Found {data['count']} models:")
        for m in data['models'][:5]:
            print_info(f"  - ID {m['id']}: {m['name']}")
        if data['count'] > 5:
            print_info(f"  ... and {data['count'] - 5} more")
        
        print_success("Models listing passed")
        return True
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_interact_rounds():
    """Test multiple interaction rounds."""
    print_header("Test 3: Interaction Flow (3 rounds)")
    
    session_id = None
    
    # Round 1: First interaction (no vote)
    print("\n  Round 1: Initial prompt (no previous vote)")
    try:
        response = requests.post(f"{BASE_URL}/interact", json={
            "prompt": "What is artificial intelligence?",
            "previous_vote": None,
            "feedback_text": ""
        }, timeout=120)
        
        if response.status_code != 200:
            print_error(f"Failed with status {response.status_code}")
            print_info(response.text)
            return False
            
        data = response.json()
        session_id = data["session_id"]
        
        print_info(f"Session: {session_id[:8]}...")
        print_info(f"Round: {data['round']}")
        print_info(f"CUPID: {data['cLeft']['model_name']} vs {data['cRight']['model_name']}")
        print_info(f"Baseline: {data['bLeft']['model_name']} vs {data['bRight']['model_name']}")
        
        # Show response previews
        print_info(f"CUPID Left response: {data['cLeft']['text'][:80]}...")
        
        print_success("Round 1 completed")
        
    except Exception as e:
        print_error(f"Round 1 failed: {e}")
        return False
    
    # Round 2: Vote for CUPID left
    print("\n  Round 2: After voting 'cupid_left'")
    try:
        response = requests.post(f"{BASE_URL}/interact", json={
            "session_id": session_id,
            "prompt": "Explain machine learning briefly.",
            "previous_vote": "cupid_left",
            "feedback_text": "I prefer faster models"
        }, timeout=120)
        
        data = response.json()
        print_info(f"Round: {data['round']}")
        print_info(f"CUPID: {data['cLeft']['model_name']} vs {data['cRight']['model_name']}")
        print_info(f"Baseline: {data['bLeft']['model_name']} vs {data['bRight']['model_name']}")
        
        print_success("Round 2 completed")
        
    except Exception as e:
        print_error(f"Round 2 failed: {e}")
        return False
    
    # Round 3: Vote for Baseline right
    print("\n  Round 3: After voting 'baseline_right'")
    try:
        response = requests.post(f"{BASE_URL}/interact", json={
            "session_id": session_id,
            "prompt": "What is deep learning?",
            "previous_vote": "baseline_right",
            "feedback_text": "I need cheaper options"
        }, timeout=120)
        
        data = response.json()
        print_info(f"Round: {data['round']}")
        print_info(f"CUPID: {data['cLeft']['model_name']} vs {data['cRight']['model_name']}")
        print_info(f"Baseline: {data['bLeft']['model_name']} vs {data['bRight']['model_name']}")
        
        print_success("Round 3 completed")
        
    except Exception as e:
        print_error(f"Round 3 failed: {e}")
        return False
    
    return session_id


def test_session_info(session_id):
    """Test session info endpoint."""
    print_header("Test 4: Session Info")
    try:
        response = requests.get(f"{BASE_URL}/session/{session_id}", timeout=10)
        data = response.json()
        
        print_info(f"Session ID: {data['session_id'][:8]}...")
        print_info(f"Total rounds: {data['round_count']}")
        print_info(f"CUPID rounds: {data['cupid_rounds']}")
        print_info(f"Baseline rounds: {data['baseline_rounds']}")
        print_info(f"Num models: {data['num_models']}")
        
        print_success("Session info passed")
        return True
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def test_delete_session(session_id):
    """Test session deletion."""
    print_header("Test 5: Delete Session")
    try:
        response = requests.delete(f"{BASE_URL}/session/{session_id}", timeout=10)
        
        if response.status_code == 200:
            print_success("Session deleted successfully")
            
            # Verify it's deleted
            verify = requests.get(f"{BASE_URL}/session/{session_id}", timeout=10)
            if verify.status_code == 404:
                print_success("Verified: session no longer exists")
            return True
        else:
            print_error(f"Failed with status {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Failed: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  USER STUDY BACKEND - API TEST SUITE")
    print("=" * 60)
    print(f"\nServer URL: {BASE_URL}")
    print("Starting tests...\n")
    
    start_time = time.time()
    
    # Run tests
    results = []
    
    # Test 1: Health
    results.append(("Health Check", test_health()))
    if not results[-1][1]:
        print("\n❌ Server not available. Stopping tests.")
        return
    
    # Test 2: Models
    results.append(("List Models", test_models()))
    
    # Test 3: Interactions (returns session_id or False)
    session_id = test_interact_rounds()
    results.append(("Interaction Flow", bool(session_id)))
    
    if session_id:
        # Test 4: Session info
        results.append(("Session Info", test_session_info(session_id)))
        
        # Test 5: Delete session
        results.append(("Delete Session", test_delete_session(session_id)))
    
    # Summary
    elapsed = time.time() - start_time
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    print(f"\n  Results: {passed}/{total} tests passed")
    print(f"  Time: {elapsed:.2f} seconds")
    
    if passed == total:
        print("\n  🎉 All tests passed! Backend is working correctly.")
    else:
        print("\n  ⚠️  Some tests failed. Check the output above.")
    
    print()


if __name__ == "__main__":
    main()
