#!/usr/bin/env python3
"""
Complete Travel Planning Conversation Test Script
Testing Memory System's Query Response Capability
"""

import sys
import os
import json
import time
sys.path.append('.')

from memoryos import Memoryos

def main():
    print("=" * 60)
    print("🚀 MemoryOS Travel Planning Memory Test")
    print("=" * 60)
    
    # Create Memoryos instance
    memoryos = Memoryos(
        user_id='travel_user_test',
        openai_api_key='',
        openai_base_url='https://cn2us02.opapi.win/v1',
        data_storage_path='./comprehensive_test_data',
        assistant_id='travel_assistant',
        embedding_model_name='',
        mid_term_capacity=1000,
        mid_term_heat_threshold=12.0,
        mid_term_similarity_threshold=0.7,
        short_term_capacity=2,
        llm_model='gpt-4.1-mini'
    )
    
    print("📝 Phase 1: Adding 30 rounds of travel planning conversations...")
    
    # 30 rounds of rich travel planning conversations in English
    conversations = [
        # Basic information and travel preferences (1-10)
        ("Hello, I want to plan a trip", "Hello! I'd be happy to help you plan your trip. Where would you like to travel?"),
        ("My name is Emily, I'm 28 years old, and I'm a graphic designer", "Nice to meet you, Emily! As a graphic designer, you must have great aesthetic taste."),
        ("I prefer artistic and cultural travel destinations", "Artistic places are very charming! Do you prefer historical culture or modern art?"),
        ("I prefer historical culture, ancient architecture and museums", "Historical culture is very enriching! Have you considered European or Asian ancient cities?"),
        ("I want to go to Japan, especially Kyoto and Nara", "Japan's ancient capitals are beautiful! Kyoto's temples and Nara's deer are very famous."),
        ("I'm planning to go in October for about 7-10 days", "October is the best season to visit Japan! The autumn foliage season is beautiful."),
        ("My budget is around 15,000-20,000 yuan", "That's a reasonable budget! We can arrange a very nice itinerary."),
        ("I prefer niche places, don't want to go to overly commercialized spots", "I understand! You prefer experiencing local culture rather than tourist hotspots."),
        ("Yes, I hope to deeply experience local life", "Deep travel is very meaningful! Would you like to try staying at a guesthouse or traditional inn?"),
        ("I want to experience traditional Japanese ryokan", "Japanese ryokan are very special! Hot springs and kaiseki cuisine are worth experiencing."),
        
        # Specific itinerary planning (11-20)
        ("Can you recommend some niche temples in Kyoto?", "I recommend Sanzen-in, Jakko-in, and Tenryu-ji in Arashiyama."),
        ("I'm very interested in traditional handicrafts", "Kyoto has many traditional crafts! Kiyomizu pottery and Nishijin weaving are very famous."),
        ("Can you arrange some handicraft experience courses?", "Of course! Pottery, kimono making, and tea ceremony all have experience courses."),
        ("I want to learn tea ceremony, do I need to make a reservation?", "Tea ceremony experiences need advance booking, let me check available times for you."),
        ("Besides Kyoto, what do you recommend in Nara?", "Nara's Todai-ji Temple, Kasuga Taisha Shrine are worth seeing, plus Deer Park."),
        ("I heard Nara has many alleys with great atmosphere", "Yes! Naramachi's old streets have great charm, with many traditional shops."),
        ("How should I arrange transportation?", "I suggest buying a JR Kansai Area Pass, transportation between Kyoto and Nara is very convenient."),
        ("I want to go to some restaurants recommended by locals", "I recommend some local shops not in tourist guides, very authentic!"),
        ("Do you have vegetarian restaurant recommendations? I pay attention to healthy eating", "Kyoto has many shojin ryori restaurants, both healthy and cultural experience."),
        ("I want to learn about local festival culture", "In October there's Jidai Matsuri, one of Kyoto's three major festivals, very worth seeing!"),
        
        # In-depth needs and personal preferences (21-30)
        ("I enjoy photography, what are some good photo spots?", "Bamboo Grove, Fushimi Inari's thousands of torii gates are perfect for photography!"),
        ("I especially like photographing architecture and people", "You'll definitely love Kinkaku-ji's reflection and the geisha district streetscapes."),
        ("I don't like crowded places", "I recommend some early morning time slots, fewer tourists and great lighting."),
        ("I want to buy some traditional crafts as souvenirs", "Nishijin weaving items and Kiyomizu pottery tea sets have great collectible value."),
        ("Are there any seasonal experience activities?", "In October you can participate in momiji-gari (autumn leaf viewing) and hot spring bathing while viewing maples."),
        ("I want to try some local lifestyle experiences", "We can arrange early morning visits to fish markets to experience locals' rhythm."),
        ("For accommodation, I hope to experience different types", "We can arrange 2 nights at traditional ryokan, others at boutique guesthouses."),
        ("I'm also interested in Japanese flower arrangement", "Kyoto has many ikebana school experience classes, we can arrange one session."),
        ("Do you have shopping suggestions?", "I recommend some long-established shops, good quality and historical significance."),
        ("Overall, I hope this trip has rich cultural content", "Understood! I'll arrange a deep cultural experience journey for you.")
    ]
    
    # Add conversations
    for i, (user_input, agent_response) in enumerate(conversations, 1):
        print(f"  [{i:2d}/{len(conversations)}] Adding conversation: {user_input[:40]}...")
        memoryos.add_memory(user_input, agent_response)
        
        # Display status every 10 rounds
        if i % 10 == 0:
            sessions = memoryos.mid_term_memory.sessions
            if sessions:
                max_heat = max(session.get('H_segment', 0) for session in sessions.values())
                print(f"    Current max heat: {max_heat:.2f}")
    
    print(f"\n🔥 Phase 2: Force triggering mid-term analysis...")
    memoryos.force_mid_term_analysis()
    
    print(f"\n⏳ Phase 3: Waiting for system synchronization...")
    time.sleep(2)
    
    print(f"\n🧠 Phase 4: Testing Memory System Query Response...")
    
    # Test queries based on previous conversations
    test_queries = [
        {
            "query": "What's my name and what's my profession?",
            "expected_keywords": ["Emily", "graphic designer", "designer"],
            "description": "Testing basic user information recall"
        },
        {
            "query": "Where do I want to travel and what are my preferences?",
            "expected_keywords": ["Japan", "Kyoto", "Nara", "historical", "culture"],
            "description": "Testing travel destination and preference recall"
        },
        {
            "query": "What are my hobbies and what kind of experiences am I interested in?",
            "expected_keywords": ["photography", "traditional", "cultural"],
            "description": "Testing hobby and interest recall"
        }
    ]
    
    print("\n" + "="*60)
    print("🔍 MEMORY SYSTEM QUERY TESTING")
    print("="*60)
    
    total_score = 0
    max_score = len(test_queries)
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}: {test_case['description']}")
        print(f"Question: {test_case['query']}")
        print("-" * 50)
        
        try:
            # Get response from memory system
            response = memoryos.get_response(test_case['query'])
            print(f"System Response: {response}")
            
            # Check if expected keywords are in the response
            response_lower = response.lower()
            found_keywords = []
            missing_keywords = []
            
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)
            
            # Calculate score for this query
            keyword_score = len(found_keywords) / len(test_case['expected_keywords'])
            
            print(f"\n✅ Found keywords: {found_keywords}")
            if missing_keywords:
                print(f"❌ Missing keywords: {missing_keywords}")
            
            print(f"🎯 Keyword match rate: {keyword_score:.1%}")
            
            # Determine if this test passed (>50% keyword match)
            if keyword_score >= 0.5:
                print(f"✅ Test {i}: PASSED")
                total_score += 1
            else:
                print(f"❌ Test {i}: FAILED")
                
        except Exception as e:
            print(f"❌ Error during query {i}: {e}")
            print(f"❌ Test {i}: FAILED")
    
    # Final results
    print("\n" + "="*60)
    print("📊 FINAL TEST RESULTS")
    print("="*60)
    
    success_rate = (total_score / max_score) * 100
    print(f"Passed Tests: {total_score}/{max_score}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Conversations Added: {len(conversations)}")
    print(f"Test Theme: Japan Travel Planning")
    print(f"User Profile: Emily, 28-year-old graphic designer, loves cultural travel and photography")
    
    if success_rate >= 70:
        print("\n🎉 EXCELLENT! Memory system performed very well!")
        return True
    elif success_rate >= 50:
        print("\n👍 GOOD! Memory system performed adequately!")
        return True
    else:
        print("\n😞 NEEDS IMPROVEMENT! Memory system needs optimization!")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 Congratulations! MemoryOS Travel Planning Memory Test Completed Successfully!")
    else:
        print("\n🔧 Memory system needs further optimization.") 