    async def _do_mid_game_analysis(self,
                                 game_type: str,
                                 moves_made: int) -> None:
        """Perform mid-game analysis and rest period.
        
        Args:
            game_type: Type of game being played
            moves_made: Number of moves made so far
        """
        try:
            # Check if we should do mid-game analysis
            if moves_made % 10 != 0:  # Every 10 moves
                return
                
            consolidation = await self.consolidation_manager.mid_game_rest(
                self.current_game_states,
                self.current_actions,
                self.current_scores,
                game_type
            )
            
            # Log insights
            if consolidation.new_patterns_discovered:
                logger.info("🔍 New patterns discovered during rest:")
                for pattern in consolidation.new_patterns_discovered[:3]:
                    logger.info(f"  - {pattern['type']}: {pattern.get('score_gain', 0):.2f} score gain")
                    
            if consolidation.confirmed_hypotheses:
                logger.info("✅ Confirmed hypotheses:")
                for hypothesis in consolidation.confirmed_hypotheses[:3]:
                    logger.info(f"  - {hypothesis['type']}")
                    
            # Apply movement optimizations
            if consolidation.movement_optimizations:
                logger.info("🔄 Movement optimizations suggested:")
                for opt in consolidation.movement_optimizations[:3]:
                    logger.info(f"  - {opt}")
                    
            # Update action values
            if consolidation.updated_action_values:
                top_actions = sorted(
                    consolidation.updated_action_values.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
                logger.info("⭐ Top performing actions:")
                for action, value in top_actions:
                    logger.info(f"  - Action {action}: {value:.2f}")
            
            # Make predictions
            if consolidation.next_state_predictions:
                logger.info(f"🎯 Predicting {len(consolidation.next_state_predictions)} possible next states")
                
        except Exception as e:
            logger.error(f"Error during mid-game analysis: {e}")
            
    async def _do_post_game_analysis(self,
                                  game_type: str,
                                  final_score: float,
                                  win: bool) -> None:
        """Perform post-game analysis and consolidation.
        
        Args:
            game_type: Type of game played
            final_score: Final game score
            win: Whether the game was won
        """
        try:
            consolidation = await self.consolidation_manager.post_game_consolidation(
                self.current_game_states,
                self.current_actions,
                self.current_scores,
                final_score,
                game_type
            )
            
            # Log comprehensive analysis
            logger.info("\n=== Post-Game Analysis ===")
            
            if consolidation.new_patterns_discovered:
                logger.info("\n🔍 Pattern Analysis:")
                for pattern in consolidation.new_patterns_discovered[:5]:
                    if pattern['type'] == 'progressive_sequence':
                        logger.info(
                            f"  Found {pattern['length']}-move sequence "
                            f"with {pattern['score_gain']:.1f} score gain "
                            f"({pattern['reliability']*100:.1f}% reliable)"
                        )
                    elif pattern['type'] == 'state_pattern':
                        logger.info(
                            f"  State pattern found with expected gain: "
                            f"{pattern['expected_gain']:.1f}"
                        )
                        
            if consolidation.confirmed_hypotheses:
                logger.info("\n✅ Confirmed Strategies:")
                for hypothesis in consolidation.confirmed_hypotheses[:5]:
                    if hypothesis['type'] == 'action_sequence':
                        logger.info(
                            f"  Action sequence {hypothesis['sequence']} "
                            f"reliably gains {hypothesis['min_score_gain']:.1f}+ score"
                        )
                        
            if consolidation.rejected_hypotheses:
                logger.info("\n❌ Disproven Strategies:")
                for hypothesis in consolidation.rejected_hypotheses[:5]:
                    logger.info(f"  {hypothesis['type']} hypothesis rejected")
                    
            # Update action values
            if consolidation.updated_action_values:
                logger.info("\n⭐ Final Action Rankings:")
                rankings = sorted(
                    consolidation.updated_action_values.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                for action, value in rankings[:5]:
                    logger.info(f"  Action {action}: {value:.2f}")
                    
            if consolidation.movement_optimizations:
                logger.info("\n🔄 Movement Optimization Tips:")
                for tip in consolidation.movement_optimizations[:5]:
                    logger.info(f"  - {tip}")
                    
            # Reset game state tracking
            self.current_game_states = []
            self.current_actions = []
            self.current_scores = []
            
        except Exception as e:
            logger.error(f"Error during post-game analysis: {e}")
            