#!/usr/bin/env python3
"""
IA2 STRING INDICES ERROR DEBUGGING TEST SUITE
FORCE IA2 STRATEGIC REASONING ERROR DEBUGGING

Specifically focus on triggering the IA2 "string indices must be integers, not 'str'" error to debug and fix it.

OBJECTIVES:
1. **Trigger IA2 Execution**: Force multiple IA1→IA2 escalations with high confidence/RR symbols
2. **Catch IA2 Error**: Monitor for the "string indices" error in IA2 strategic reasoning
3. **Identify Exact Location**: Pinpoint where in the IA2 code the error occurs
4. **Debug the Error**: Show the exact traceback and problematic code line

TESTING APPROACH:
- Force run IA1 analysis for multiple symbols: BTCUSDT, ETHUSDT, SOLUSDT
- Ensure IA1 confidence ≥70% or RR ≥2.0 to trigger IA2 escalation
- Monitor backend logs for IA2 strategic prompt execution
- Catch and analyze the "string indices must be integers, not 'str'" error
- Show the exact JSON parsing or f-string interpolation that fails

FOCUS AREAS:
- IA2 strategic prompt generation
- Claude response parsing
- JSON.loads() operations in IA2
- Any f-string interpolations with dictionary access
- Enhanced IA2 strategic fields parsing

The goal is to catch the exact moment and location where IA2 crashes with the "string indices" error so we can fix it permanently.
"""

import asyncio
import json
import logging
import os
import sys
import time
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List
import requests
import subprocess
from pymongo import MongoClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IA2StringIndicesErrorDebugger:
    """Specialized test suite to trigger and debug IA2 string indices error"""
    
    def __init__(self):
        # Get backend URL from frontend env
        try:
            with open('/app/frontend/.env', 'r') as f:
                for line in f:
                    if line.startswith('REACT_APP_BACKEND_URL='):
                        backend_url = line.split('=')[1].strip()
                        break
                else:
                    backend_url = "http://localhost:8001"
        except Exception:
            backend_url = "http://localhost:8001"
        
        self.api_url = f"{backend_url}/api"
        logger.info(f"🎯 IA2 String Indices Error Debugging at: {self.api_url}")
        
        # MongoDB connection for direct database analysis
        try:
            self.mongo_client = MongoClient("mongodb://localhost:27017")
            self.db = self.mongo_client["myapp"]
            logger.info("✅ MongoDB connection established for error debugging")
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            self.mongo_client = None
            self.db = None
        
        # Target symbols for IA2 escalation
        self.target_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT"]
        
        # Error tracking
        self.ia2_executions = []
        self.string_indices_errors = []
        self.backend_log_analysis = {}
        
    def log_debug_info(self, message: str, level: str = "INFO"):
        """Log debug information with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if level == "ERROR":
            logger.error(f"🚨 [{timestamp}] {message}")
        elif level == "WARNING":
            logger.warning(f"⚠️ [{timestamp}] {message}")
        else:
            logger.info(f"🔍 [{timestamp}] {message}")
    
    async def force_ia1_analysis_for_ia2_escalation(self, symbol: str) -> Dict[str, Any]:
        """Force IA1 analysis specifically to trigger IA2 escalation"""
        self.log_debug_info(f"Forcing IA1 analysis for {symbol} to trigger IA2 escalation")
        
        try:
            # Force IA1 analysis
            response = requests.post(f"{self.api_url}/force-ia1-analysis", 
                                   json={"symbol": symbol}, 
                                   timeout=120)
            
            if response.status_code in [200, 201]:
                result = response.json()
                
                confidence = result.get('confidence', 0)
                rr = result.get('risk_reward_ratio', 0)
                signal = result.get('recommendation', 'UNKNOWN')
                
                # Determine if IA2 escalation should occur
                should_escalate = False
                voie_path = None
                
                if signal.upper() in ['LONG', 'SHORT']:
                    if confidence >= 0.95:
                        should_escalate = True
                        voie_path = "VOIE 3 (≥95% confidence override)"
                    elif confidence >= 0.70:
                        should_escalate = True
                        voie_path = "VOIE 1 (≥70% confidence)"
                    elif rr >= 2.0:
                        should_escalate = True
                        voie_path = "VOIE 2 (≥2.0 RR)"
                
                execution_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'ia1_success': result.get('success', False),
                    'confidence': confidence,
                    'risk_reward_ratio': rr,
                    'signal': signal,
                    'should_escalate_to_ia2': should_escalate,
                    'voie_path': voie_path,
                    'ia1_analysis_id': result.get('analysis_id'),
                    'ia2_triggered': False,
                    'ia2_error_detected': False,
                    'string_indices_error': False
                }
                
                self.log_debug_info(f"{symbol} IA1 Result: Conf={confidence:.1%}, RR={rr:.1f}, Signal={signal}")
                if should_escalate:
                    self.log_debug_info(f"{symbol} SHOULD ESCALATE to IA2 via {voie_path}")
                else:
                    self.log_debug_info(f"{symbol} will NOT escalate to IA2 (insufficient criteria)")
                
                return execution_data
                
            else:
                self.log_debug_info(f"{symbol} IA1 analysis failed: HTTP {response.status_code}", "ERROR")
                return {
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'ia1_success': False,
                    'error': f"HTTP {response.status_code}",
                    'should_escalate_to_ia2': False
                }
                
        except Exception as e:
            self.log_debug_info(f"{symbol} IA1 analysis exception: {str(e)}", "ERROR")
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'ia1_success': False,
                'error': str(e),
                'should_escalate_to_ia2': False
            }
    
    async def monitor_backend_logs_for_ia2_errors(self, duration_seconds: int = 180):
        """Monitor backend logs specifically for IA2 string indices errors"""
        self.log_debug_info(f"Starting backend log monitoring for {duration_seconds} seconds")
        
        log_files = [
            "/var/log/supervisor/backend.out.log",
            "/var/log/supervisor/backend.err.log"
        ]
        
        # Get initial log positions
        initial_positions = {}
        for log_file in log_files:
            try:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        f.seek(0, 2)  # Go to end of file
                        initial_positions[log_file] = f.tell()
                else:
                    initial_positions[log_file] = 0
            except Exception as e:
                self.log_debug_info(f"Could not get initial position for {log_file}: {e}", "WARNING")
                initial_positions[log_file] = 0
        
        # Monitor for new log entries
        start_time = time.time()
        ia2_activity_detected = False
        string_indices_errors_found = []
        
        while time.time() - start_time < duration_seconds:
            for log_file in log_files:
                try:
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            f.seek(initial_positions[log_file])
                            new_content = f.read()
                            
                            if new_content:
                                lines = new_content.split('\n')
                                
                                for line in lines:
                                    if line.strip():
                                        # Check for IA2 activity
                                        if any(keyword in line for keyword in ['IA2', 'claude', 'strategic', 'make_decision']):
                                            if not ia2_activity_detected:
                                                self.log_debug_info("🚀 IA2 ACTIVITY DETECTED in logs!")
                                                ia2_activity_detected = True
                                            
                                            # Log IA2 activity
                                            self.log_debug_info(f"IA2 LOG: {line.strip()}")
                                        
                                        # Check for string indices error
                                        if 'string indices must be integers' in line.lower():
                                            error_info = {
                                                'timestamp': datetime.now().isoformat(),
                                                'log_file': log_file,
                                                'error_line': line.strip(),
                                                'context_lines': []
                                            }
                                            
                                            # Get context lines around the error
                                            line_index = lines.index(line)
                                            start_context = max(0, line_index - 5)
                                            end_context = min(len(lines), line_index + 5)
                                            
                                            for i in range(start_context, end_context):
                                                if i < len(lines) and lines[i].strip():
                                                    error_info['context_lines'].append(f"[{i-line_index:+d}] {lines[i].strip()}")
                                            
                                            string_indices_errors_found.append(error_info)
                                            self.log_debug_info("🚨 STRING INDICES ERROR DETECTED!", "ERROR")
                                            self.log_debug_info(f"Error line: {line.strip()}", "ERROR")
                                            
                                            # Log context
                                            self.log_debug_info("📋 ERROR CONTEXT:", "ERROR")
                                            for context_line in error_info['context_lines']:
                                                self.log_debug_info(f"   {context_line}", "ERROR")
                                
                                # Update position
                                initial_positions[log_file] = f.tell()
                                
                except Exception as e:
                    self.log_debug_info(f"Error reading {log_file}: {e}", "WARNING")
            
            # Short sleep to avoid excessive CPU usage
            await asyncio.sleep(2)
        
        self.log_debug_info(f"Backend log monitoring completed. IA2 activity: {ia2_activity_detected}, String errors: {len(string_indices_errors_found)}")
        
        return {
            'ia2_activity_detected': ia2_activity_detected,
            'string_indices_errors': string_indices_errors_found,
            'monitoring_duration': duration_seconds
        }
    
    async def analyze_ia2_database_entries(self):
        """Analyze recent IA2 database entries for error patterns"""
        self.log_debug_info("Analyzing IA2 database entries for error patterns")
        
        if self.db is None:
            self.log_debug_info("MongoDB not available for database analysis", "WARNING")
            return {'error': 'MongoDB not available'}
        
        try:
            # Get recent IA2 decisions
            recent_decisions = list(self.db.trading_decisions.find({}).sort("timestamp", -1).limit(20))
            
            analysis = {
                'total_decisions': len(recent_decisions),
                'decisions_with_errors': 0,
                'missing_strategic_fields': 0,
                'incomplete_reasoning': 0,
                'error_patterns': [],
                'field_analysis': {
                    'calculated_rr': 0,
                    'rr_reasoning': 0,
                    'strategic_reasoning': 0,
                    'market_regime_assessment': 0,
                    'position_size_recommendation': 0,
                    'execution_priority': 0
                }
            }
            
            self.log_debug_info(f"Analyzing {len(recent_decisions)} recent IA2 decisions")
            
            for decision in recent_decisions:
                # Check for error indicators
                reasoning = decision.get('reasoning', '').lower()
                
                if any(error_keyword in reasoning for error_keyword in ['error', 'exception', 'failed', 'string indices']):
                    analysis['decisions_with_errors'] += 1
                    analysis['error_patterns'].append({
                        'symbol': decision.get('symbol', 'UNKNOWN'),
                        'timestamp': decision.get('timestamp', 'UNKNOWN'),
                        'error_snippet': reasoning[:200] + '...' if len(reasoning) > 200 else reasoning
                    })
                
                # Check for missing strategic fields
                strategic_fields = ['calculated_rr', 'rr_reasoning', 'strategic_reasoning', 
                                  'market_regime_assessment', 'position_size_recommendation', 'execution_priority']
                
                missing_fields = 0
                for field in strategic_fields:
                    if field in decision and decision[field] is not None:
                        analysis['field_analysis'][field] += 1
                    else:
                        missing_fields += 1
                
                if missing_fields >= len(strategic_fields) * 0.5:  # More than 50% fields missing
                    analysis['missing_strategic_fields'] += 1
                
                # Check reasoning quality
                if len(reasoning) < 50:  # Very short reasoning
                    analysis['incomplete_reasoning'] += 1
            
            # Log analysis results
            self.log_debug_info("📊 IA2 DATABASE ANALYSIS RESULTS:")
            self.log_debug_info(f"   Total decisions analyzed: {analysis['total_decisions']}")
            self.log_debug_info(f"   Decisions with error indicators: {analysis['decisions_with_errors']}")
            self.log_debug_info(f"   Missing strategic fields: {analysis['missing_strategic_fields']}")
            self.log_debug_info(f"   Incomplete reasoning: {analysis['incomplete_reasoning']}")
            
            self.log_debug_info("📊 STRATEGIC FIELD COVERAGE:")
            for field, count in analysis['field_analysis'].items():
                coverage = count / analysis['total_decisions'] if analysis['total_decisions'] > 0 else 0
                self.log_debug_info(f"   {field}: {count}/{analysis['total_decisions']} ({coverage:.1%})")
            
            if analysis['error_patterns']:
                self.log_debug_info("🚨 ERROR PATTERNS FOUND:")
                for i, error in enumerate(analysis['error_patterns'][:5]):  # Show first 5 errors
                    self.log_debug_info(f"   Error {i+1}: {error['symbol']} - {error['error_snippet']}")
            
            return analysis
            
        except Exception as e:
            self.log_debug_info(f"Database analysis failed: {str(e)}", "ERROR")
            return {'error': str(e)}
    
    async def trigger_ia2_string_indices_error_debugging(self):
        """Main function to trigger and debug IA2 string indices error"""
        self.log_debug_info("🚀 STARTING IA2 STRING INDICES ERROR DEBUGGING")
        self.log_debug_info("=" * 80)
        
        # Step 1: Start backend log monitoring
        self.log_debug_info("STEP 1: Starting backend log monitoring")
        log_monitor_task = asyncio.create_task(self.monitor_backend_logs_for_ia2_errors(300))  # 5 minutes
        
        # Small delay to ensure monitoring is active
        await asyncio.sleep(5)
        
        # Step 2: Force multiple IA1 analyses to trigger IA2 escalations
        self.log_debug_info("STEP 2: Forcing IA1 analyses to trigger IA2 escalations")
        
        for symbol in self.target_symbols:
            self.log_debug_info(f"🎯 Processing {symbol} for IA2 escalation")
            
            execution_result = await self.force_ia1_analysis_for_ia2_escalation(symbol)
            self.ia2_executions.append(execution_result)
            
            # Wait between requests to allow processing
            await asyncio.sleep(10)
        
        # Step 3: Wait for IA2 processing and log monitoring
        self.log_debug_info("STEP 3: Waiting for IA2 processing to complete")
        self.log_debug_info("⏳ Monitoring backend logs for IA2 activity and string indices errors...")
        
        # Wait for log monitoring to complete
        log_analysis = await log_monitor_task
        
        # Step 4: Analyze database for IA2 entries and errors
        self.log_debug_info("STEP 4: Analyzing IA2 database entries")
        db_analysis = await self.analyze_ia2_database_entries()
        
        # Step 5: Compile comprehensive debugging report
        self.log_debug_info("STEP 5: Compiling comprehensive debugging report")
        
        # Analyze IA1→IA2 escalation success
        should_escalate_count = sum(1 for exec in self.ia2_executions if exec.get('should_escalate_to_ia2', False))
        successful_ia1_count = sum(1 for exec in self.ia2_executions if exec.get('ia1_success', False))
        
        self.log_debug_info("=" * 80)
        self.log_debug_info("📊 IA2 STRING INDICES ERROR DEBUGGING REPORT")
        self.log_debug_info("=" * 80)
        
        self.log_debug_info("🎯 IA1→IA2 ESCALATION ANALYSIS:")
        self.log_debug_info(f"   Symbols processed: {len(self.target_symbols)}")
        self.log_debug_info(f"   Successful IA1 analyses: {successful_ia1_count}")
        self.log_debug_info(f"   Should escalate to IA2: {should_escalate_count}")
        
        # Show escalation details
        for execution in self.ia2_executions:
            if execution.get('should_escalate_to_ia2', False):
                symbol = execution['symbol']
                voie = execution.get('voie_path', 'Unknown')
                conf = execution.get('confidence', 0)
                rr = execution.get('risk_reward_ratio', 0)
                self.log_debug_info(f"   ✅ {symbol}: {voie} (Conf: {conf:.1%}, RR: {rr:.1f})")
        
        self.log_debug_info("🔍 BACKEND LOG MONITORING RESULTS:")
        self.log_debug_info(f"   IA2 activity detected: {log_analysis['ia2_activity_detected']}")
        self.log_debug_info(f"   String indices errors found: {len(log_analysis['string_indices_errors'])}")
        
        if log_analysis['string_indices_errors']:
            self.log_debug_info("🚨 STRING INDICES ERRORS DETECTED:")
            for i, error in enumerate(log_analysis['string_indices_errors']):
                self.log_debug_info(f"   Error {i+1}:")
                self.log_debug_info(f"     File: {error['log_file']}")
                self.log_debug_info(f"     Time: {error['timestamp']}")
                self.log_debug_info(f"     Line: {error['error_line']}")
                self.log_debug_info(f"     Context:")
                for context_line in error['context_lines']:
                    self.log_debug_info(f"       {context_line}")
        
        self.log_debug_info("📊 DATABASE ANALYSIS RESULTS:")
        if 'error' not in db_analysis:
            self.log_debug_info(f"   Total IA2 decisions: {db_analysis['total_decisions']}")
            self.log_debug_info(f"   Decisions with errors: {db_analysis['decisions_with_errors']}")
            self.log_debug_info(f"   Missing strategic fields: {db_analysis['missing_strategic_fields']}")
            self.log_debug_info(f"   Incomplete reasoning: {db_analysis['incomplete_reasoning']}")
            
            # Strategic field coverage
            self.log_debug_info("   Strategic field coverage:")
            for field, count in db_analysis['field_analysis'].items():
                coverage = count / db_analysis['total_decisions'] if db_analysis['total_decisions'] > 0 else 0
                self.log_debug_info(f"     {field}: {coverage:.1%}")
        else:
            self.log_debug_info(f"   Database analysis error: {db_analysis['error']}")
        
        # Final verdict
        self.log_debug_info("=" * 80)
        self.log_debug_info("🏆 DEBUGGING VERDICT:")
        
        if log_analysis['string_indices_errors']:
            self.log_debug_info("✅ SUCCESS: STRING INDICES ERROR SUCCESSFULLY TRIGGERED AND CAPTURED!")
            self.log_debug_info(f"   Found {len(log_analysis['string_indices_errors'])} string indices errors")
            self.log_debug_info("   Error location and context captured for debugging")
            self.log_debug_info("   Ready for code analysis and fix implementation")
        elif log_analysis['ia2_activity_detected']:
            self.log_debug_info("⚠️ PARTIAL SUCCESS: IA2 activity detected but no string indices errors")
            self.log_debug_info("   IA2 may be working correctly or error may be intermittent")
            self.log_debug_info("   Consider running more test cycles or different symbols")
        elif should_escalate_count > 0:
            self.log_debug_info("⚠️ ESCALATION READY: IA1 analyses should trigger IA2 but no IA2 activity detected")
            self.log_debug_info("   Check if IA2 escalation logic is working correctly")
            self.log_debug_info("   Verify VOIE escalation paths are implemented")
        else:
            self.log_debug_info("❌ NO ESCALATION: IA1 analyses did not meet IA2 escalation criteria")
            self.log_debug_info("   Try different symbols or adjust confidence/RR thresholds")
            self.log_debug_info("   Verify IA1 analysis is generating high confidence results")
        
        self.log_debug_info("=" * 80)
        
        return {
            'ia1_executions': self.ia2_executions,
            'log_analysis': log_analysis,
            'db_analysis': db_analysis,
            'should_escalate_count': should_escalate_count,
            'successful_ia1_count': successful_ia1_count,
            'string_indices_errors_found': len(log_analysis['string_indices_errors'])
        }

async def main():
    """Main function to run IA2 string indices error debugging"""
    debugger = IA2StringIndicesErrorDebugger()
    
    try:
        results = await debugger.trigger_ia2_string_indices_error_debugging()
        
        # Exit codes based on results
        if results['string_indices_errors_found'] > 0:
            logger.info("🎯 SUCCESS: String indices error captured for debugging")
            sys.exit(0)  # Success - error captured
        elif results['should_escalate_count'] > 0:
            logger.info("⚠️ PARTIAL: IA2 escalation conditions met but error not triggered")
            sys.exit(1)  # Partial success
        else:
            logger.info("❌ FAILED: Could not trigger IA2 escalation")
            sys.exit(2)  # Failed to trigger
            
    except Exception as e:
        logger.error(f"🚨 DEBUGGING FAILED: {str(e)}")
        sys.exit(3)  # Exception occurred

if __name__ == "__main__":
    asyncio.run(main())