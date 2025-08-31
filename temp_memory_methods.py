    async def _load_prioritized_memories(self) -> List[Dict[str, Any]]:
        """Load and prioritize existing memories based on relevance and recency."""
        try:
            import os
            import json
            from pathlib import Path
            
            memories = []
            
            # Look for memory files in the continuous learning data directory
            memory_dirs = [
                Path("continuous_learning_data"),
                Path("meta_learning_data"), 
                Path("test_meta_learning_data")
            ]
            
            for memory_dir in memory_dirs:
                if memory_dir.exists():
                    for memory_file in memory_dir.glob("*.json"):
                        try:
                            with open(memory_file, 'r') as f:
                                memory_data = json.load(f)
                                # Add metadata
                                memory_data['file_path'] = str(memory_file)
                                memory_data['file_age'] = os.path.getmtime(memory_file)
                                memories.append(memory_data)
                        except Exception as e:
                            print(f"   ⚠️ Failed to load {memory_file}: {e}")
            
            # Prioritize memories by relevance (recent + effective)
            def priority_score(memory):
                recency_score = memory.get('file_age', 0) / 1000000  # Recent files score higher
                effectiveness_score = memory.get('final_score', 0) * 10  # Effective memories score higher
                return recency_score + effectiveness_score
            
            memories.sort(key=priority_score, reverse=True)
            
            # Return top 10 most relevant memories
            return memories[:10]
            
        except Exception as e:
            print(f"   ⚠️ Error loading memories: {e}")
            return []

    async def _garbage_collect_memories(self) -> int:
        """Delete irrelevant, old, or low-value memories to free up space."""
        try:
            import os
            import time
            from pathlib import Path
            
            deleted_count = 0
            current_time = time.time()
            
            memory_dirs = [
                Path("continuous_learning_data"),
                Path("meta_learning_data"),
                Path("test_meta_learning_data") 
            ]
            
            for memory_dir in memory_dirs:
                if memory_dir.exists():
                    for memory_file in memory_dir.glob("*.json"):
                        try:
                            file_age_days = (current_time - os.path.getmtime(memory_file)) / 86400
                            file_size_kb = os.path.getsize(memory_file) / 1024
                            
                            # Delete if: very old (>7 days) OR very small (<1KB) OR temp files
                            should_delete = (
                                file_age_days > 7 or
                                file_size_kb < 1 or
                                'temp_' in memory_file.name or
                                'test_' in memory_file.name
                            )
                            
                            if should_delete:
                                os.remove(memory_file)
                                deleted_count += 1
                                print(f"   🗑️ Deleted {memory_file.name} (age: {file_age_days:.1f}d, size: {file_size_kb:.1f}KB)")
                                
                        except Exception as e:
                            print(f"   ⚠️ Failed to process {memory_file}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"   ⚠️ Error during garbage collection: {e}")
            return 0

    async def _combine_similar_memories(self) -> int:
        """Combine similar memory patterns to reduce redundancy."""
        try:
            import json
            import time
            import os
            from pathlib import Path
            from collections import defaultdict
            
            combined_count = 0
            memory_clusters = defaultdict(list)
            
            # Group memories by similarity patterns
            memory_dir = Path("continuous_learning_data")
            if memory_dir.exists():
                for memory_file in memory_dir.glob("*.json"):
                    try:
                        with open(memory_file, 'r') as f:
                            memory_data = json.load(f)
                            
                            # Cluster by game patterns and scores
                            cluster_key = f"score_{memory_data.get('final_score', 0)//10}_actions_{memory_data.get('actions_taken', 0)//50}"
                            memory_clusters[cluster_key].append({
                                'file': memory_file,
                                'data': memory_data
                            })
                    except:
                        continue
            
            # Combine clusters with multiple similar memories
            for cluster_key, memories in memory_clusters.items():
                if len(memories) > 2:  # Combine if 3+ similar memories
                    # Create combined memory
                    combined_memory = {
                        'type': 'combined_memory_cluster',
                        'cluster_key': cluster_key,
                        'original_count': len(memories),
                        'combined_timestamp': time.time(),
                        'combined_data': {
                            'avg_score': sum(m['data'].get('final_score', 0) for m in memories) / len(memories),
                            'avg_actions': sum(m['data'].get('actions_taken', 0) for m in memories) / len(memories),
                            'patterns': [m['data'] for m in memories[:3]]  # Keep top 3 examples
                        }
                    }
                    
                    # Save combined memory
                    combined_file = memory_dir / f"combined_{cluster_key}_{int(time.time())}.json"
                    with open(combined_file, 'w') as f:
                        json.dump(combined_memory, f, indent=2)
                    
                    # Delete original files
                    for memory in memories:
                        try:
                            os.remove(memory['file'])
                        except:
                            pass
                    
                    combined_count += 1
                    print(f"   🔗 Combined {len(memories)} memories into {combined_file.name}")
            
            return combined_count
            
        except Exception as e:
            print(f"   ⚠️ Error combining memories: {e}")
            return 0

    async def _strengthen_action_memory_with_context(self, action: Dict[str, Any], context_memories: List[Dict[str, Any]]) -> None:
        """Strengthen memory of an effective action with contextual information."""
        try:
            import time
            
            action_type = action.get('action_type', 'unknown')
            effectiveness = action.get('effectiveness', 0)
            score = action.get('score_achieved', 0)
            
            print(f"   💪 Strengthening: {action_type} (effectiveness: {effectiveness:.2f}, score: {score})")
            
            # Create enhanced memory entry
            enhanced_memory = {
                'action_type': action_type,
                'effectiveness': effectiveness,
                'score': score,
                'timestamp': time.time(),
                'context': {
                    'similar_patterns': len([m for m in context_memories if m.get('final_score', 0) > 0]),
                    'learning_context': f"Enhanced during sleep cycle with {len(context_memories)} context memories"
                },
                'strengthened': True
            }
            
            # Add to effective action memories
            if not hasattr(self, 'effective_action_memories'):
                self.effective_action_memories = []
            
            self.effective_action_memories.append(enhanced_memory)
            
            # Keep only last 50 strengthened memories
            if len(self.effective_action_memories) > 50:
                self.effective_action_memories = self.effective_action_memories[-50:]
                
        except Exception as e:
            print(f"   ⚠️ Error strengthening memory: {e}")

    async def _generate_memory_insights(self, effective_actions: List[Dict[str, Any]], memories: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from memory patterns and current effective actions."""
        insights = []
        
        try:
            # Pattern analysis from memories
            if memories:
                successful_games = [m for m in memories if m.get('final_score', 0) > 0]
                if len(successful_games) >= 2:
                    avg_successful_actions = sum(m.get('actions_taken', 0) for m in successful_games) / len(successful_games)
                    insights.append(f"Historical pattern: Successful games average {avg_successful_actions:.0f} actions")
                
                # Look for game type patterns
                game_types = [m.get('game_id', '')[:4] for m in memories if m.get('game_id')]
                if game_types:
                    most_common_type = max(set(game_types), key=game_types.count)
                    insights.append(f"Most practiced game type: {most_common_type}* patterns")
            
            # Current action effectiveness
            if effective_actions:
                action_types = [a.get('action_type', '') for a in effective_actions]
                if action_types:
                    most_effective = max(set(action_types), key=action_types.count)
                    insights.append(f"Current session: {most_effective} actions showing highest effectiveness")
                    
                avg_effectiveness = sum(a.get('effectiveness', 0) for a in effective_actions) / len(effective_actions)
                if avg_effectiveness > 0.7:
                    insights.append("High effectiveness session - patterns worth reinforcing")
            
            # Cross-reference current actions with memory patterns
            if effective_actions and memories:
                insights.append("Memory consolidation: Linking current learnings with historical patterns")
                
        except Exception as e:
            print(f"   ⚠️ Error generating insights: {e}")
        
        return insights

    async def _prepare_memory_guidance(self, memories: List[Dict[str, Any]], effective_actions: List[Dict[str, Any]]) -> List[str]:
        """Prepare memory-informed guidance for the next game session."""
        guidance = []
        
        try:
            # Guidance from historical successes
            successful_memories = [m for m in memories if m.get('final_score', 0) > 0]
            if successful_memories:
                guidance.append(f"Historical success: Focus on patterns from {len(successful_memories)} successful games")
                
                # Extract successful action patterns
                if hasattr(self, 'effective_action_memories') and self.effective_action_memories:
                    top_actions = sorted(self.effective_action_memories, key=lambda x: x.get('effectiveness', 0), reverse=True)[:3]
                    for action in top_actions:
                        guidance.append(f"Prioritize: {action.get('action_type', 'unknown')} actions (proven effective)")
            
            # Guidance from current session
            if effective_actions:
                guidance.append(f"Current learnings: {len(effective_actions)} effective action patterns identified")
                
            # Energy and complexity guidance
            guidance.append("Energy management: Monitor action effectiveness to optimize sleep timing")
            
        except Exception as e:
            print(f"   ⚠️ Error preparing guidance: {e}")
        
        return guidance
