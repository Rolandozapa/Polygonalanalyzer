#!/usr/bin/env python3

import requests
import json

def debug_ia2_confidence():
    """Debug IA2 confidence calculation issues"""
    
    try:
        # Get decisions
        response = requests.get("https://dualtrade-ai.preview.emergentagent.com/api/decisions", timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to get decisions: {response.status_code}")
            return
        
        data = response.json()
        decisions = data.get('decisions', [])
        
        if not decisions:
            print("❌ No decisions found")
            return
        
        print(f"🔍 Analyzing {len(decisions)} decisions for confidence issues...")
        
        # Analyze first few decisions
        for i, decision in enumerate(decisions[:3]):
            symbol = decision.get('symbol', 'Unknown')
            confidence = decision.get('confidence', 0)
            reasoning = decision.get('ia2_reasoning', '')
            signal = decision.get('signal', 'hold')
            
            print(f"\n📊 Decision {i+1} - {symbol}:")
            print(f"   Signal: {signal}")
            print(f"   Final Confidence: {confidence:.3f}")
            
            # Try to extract confidence info from reasoning
            if reasoning:
                try:
                    # Look for confidence mentions in reasoning
                    if 'confidence' in reasoning.lower():
                        lines = reasoning.split('\n')
                        for line in lines:
                            if 'confidence' in line.lower():
                                print(f"   Reasoning: {line.strip()}")
                                break
                    
                    # Try to parse JSON if it's there
                    if reasoning.startswith('{'):
                        try:
                            parsed = json.loads(reasoning)
                            if 'confidence' in parsed:
                                print(f"   LLM Confidence: {parsed['confidence']}")
                            if 'data_confidence_assessment' in parsed:
                                print(f"   Data Assessment: {parsed['data_confidence_assessment']}")
                        except:
                            pass
                            
                except Exception as e:
                    print(f"   Error parsing reasoning: {e}")
        
        # Get analyses to check base confidence
        print(f"\n🔍 Checking IA1 analysis confidence...")
        response = requests.get("https://dualtrade-ai.preview.emergentagent.com/api/analyses", timeout=10)
        if response.status_code == 200:
            analyses_data = response.json()
            analyses = analyses_data.get('analyses', [])
            
            if analyses:
                for i, analysis in enumerate(analyses[:3]):
                    symbol = analysis.get('symbol', 'Unknown')
                    analysis_confidence = analysis.get('analysis_confidence', 0)
                    print(f"   Analysis {i+1} - {symbol}: IA1 confidence = {analysis_confidence:.3f}")
        
        # Get opportunities to check data confidence
        print(f"\n🔍 Checking market data confidence...")
        response = requests.get("https://dualtrade-ai.preview.emergentagent.com/api/opportunities", timeout=10)
        if response.status_code == 200:
            opps_data = response.json()
            opportunities = opps_data.get('opportunities', [])
            
            if opportunities:
                for i, opp in enumerate(opportunities[:3]):
                    symbol = opp.get('symbol', 'Unknown')
                    data_confidence = opp.get('data_confidence', 0)
                    print(f"   Opportunity {i+1} - {symbol}: Data confidence = {data_confidence:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_ia2_confidence()