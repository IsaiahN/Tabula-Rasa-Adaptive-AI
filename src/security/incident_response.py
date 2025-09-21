#!/usr/bin/env python3
"""
Incident Response System - Automated incident detection and response.

This module provides:
- Incident detection and classification
- Automated response workflows
- Incident tracking and management
- Response team coordination
- Post-incident analysis
"""

import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid
import asyncio

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IncidentStatus(Enum):
    """Incident status states."""
    DETECTED = "detected"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"
    FALSE_POSITIVE = "false_positive"


class ResponseAction(Enum):
    """Response actions."""
    ISOLATE_SYSTEM = "isolate_system"
    BLOCK_IP = "block_ip"
    QUARANTINE_FILE = "quarantine_file"
    RESET_CREDENTIALS = "reset_credentials"
    NOTIFY_TEAM = "notify_team"
    ESCALATE = "escalate"
    MONITOR = "monitor"
    COLLECT_EVIDENCE = "collect_evidence"
    UPDATE_SIGNATURES = "update_signatures"
    PATCH_SYSTEM = "patch_system"


@dataclass
class Incident:
    """Incident representation."""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    category: str
    source: str
    detection_time: datetime
    assigned_team: Optional[str] = None
    assigned_analyst: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    resolution_time: Optional[datetime] = None
    root_cause: Optional[str] = None
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResponseWorkflow:
    """Response workflow definition."""
    workflow_id: str
    name: str
    description: str
    triggers: List[Dict[str, Any]]
    actions: List[ResponseAction]
    conditions: List[Dict[str, Any]]
    priority: int
    enabled: bool = True


class IncidentResponseSystem:
    """
    Automated incident response system.
    
    Provides:
    - Incident detection and classification
    - Automated response workflows
    - Incident tracking and management
    - Response team coordination
    - Post-incident analysis
    """
    
    def __init__(self, max_incidents: int = 1000, auto_response: bool = True):
        self.max_incidents = max_incidents
        self.auto_response = auto_response
        self.logger = logging.getLogger(__name__)
        
        # Incident management
        self.incidents: Dict[str, Incident] = {}
        self.incident_queue: deque = deque(maxlen=max_incidents)
        self.active_incidents: Set[str] = set()
        
        # Response workflows
        self.response_workflows: Dict[str, ResponseWorkflow] = {}
        self.workflow_triggers: Dict[str, List[str]] = defaultdict(list)
        
        # Response teams
        self.response_teams: Dict[str, Dict[str, Any]] = {}
        self.team_assignments: Dict[str, str] = {}
        
        # Metrics and analytics
        self.response_metrics: Dict[str, Any] = defaultdict(int)
        self.incident_trends: Dict[str, List[float]] = defaultdict(list)
        
        # Initialize default workflows
        self._initialize_default_workflows()
    
    def _initialize_default_workflows(self):
        """Initialize default response workflows."""
        # Malware detection workflow
        malware_workflow = ResponseWorkflow(
            workflow_id="malware_detection",
            name="Malware Detection Response",
            description="Automated response to malware detection",
            triggers=[
                {"type": "threat_category", "value": "malware"},
                {"type": "severity", "value": "high"}
            ],
            actions=[
                ResponseAction.QUARANTINE_FILE,
                ResponseAction.ISOLATE_SYSTEM,
                ResponseAction.NOTIFY_TEAM,
                ResponseAction.COLLECT_EVIDENCE
            ],
            conditions=[
                {"type": "file_type", "value": "executable"},
                {"type": "confidence", "operator": ">", "value": 0.8}
            ],
            priority=1
        )
        self._add_workflow(malware_workflow)
        
        # Network intrusion workflow
        intrusion_workflow = ResponseWorkflow(
            workflow_id="network_intrusion",
            name="Network Intrusion Response",
            description="Automated response to network intrusions",
            triggers=[
                {"type": "threat_category", "value": "intrusion"},
                {"type": "severity", "value": "medium"}
            ],
            actions=[
                ResponseAction.BLOCK_IP,
                ResponseAction.MONITOR,
                ResponseAction.NOTIFY_TEAM,
                ResponseAction.COLLECT_EVIDENCE
            ],
            conditions=[
                {"type": "network_activity", "value": "suspicious"},
                {"type": "confidence", "operator": ">", "value": 0.7}
            ],
            priority=2
        )
        self._add_workflow(intrusion_workflow)
        
        # Data breach workflow
        breach_workflow = ResponseWorkflow(
            workflow_id="data_breach",
            name="Data Breach Response",
            description="Automated response to data breaches",
            triggers=[
                {"type": "threat_category", "value": "data_breach"},
                {"type": "severity", "value": "critical"}
            ],
            actions=[
                ResponseAction.ISOLATE_SYSTEM,
                ResponseAction.RESET_CREDENTIALS,
                ResponseAction.NOTIFY_TEAM,
                ResponseAction.ESCALATE,
                ResponseAction.COLLECT_EVIDENCE
            ],
            conditions=[
                {"type": "data_sensitivity", "value": "high"},
                {"type": "confidence", "operator": ">", "value": 0.9}
            ],
            priority=0  # Highest priority
        )
        self._add_workflow(breach_workflow)
    
    def _add_workflow(self, workflow: ResponseWorkflow):
        """Add a response workflow."""
        self.response_workflows[workflow.workflow_id] = workflow
        
        # Index triggers for fast lookup
        for trigger in workflow.triggers:
            trigger_key = f"{trigger['type']}:{trigger['value']}"
            self.workflow_triggers[trigger_key].append(workflow.workflow_id)
    
    def detect_incident(self, threat_data: Dict[str, Any], 
                       threat_assessment: Dict[str, Any]) -> Optional[Incident]:
        """
        Detect and create an incident from threat data.
        
        Args:
            threat_data: Raw threat data
            threat_assessment: Risk assessment results
            
        Returns:
            Created incident or None if no incident should be created
        """
        # Determine if this should be an incident
        if not self._should_create_incident(threat_data, threat_assessment):
            return None
        
        # Generate incident ID
        incident_id = str(uuid.uuid4())
        
        # Classify incident
        severity = self._classify_severity(threat_assessment)
        category = self._classify_category(threat_data)
        
        # Create incident
        incident = Incident(
            incident_id=incident_id,
            title=self._generate_incident_title(threat_data, category),
            description=self._generate_incident_description(threat_data, threat_assessment),
            severity=severity,
            status=IncidentStatus.DETECTED,
            category=category,
            source=threat_data.get("source", "unknown"),
            detection_time=datetime.now(),
            tags=self._extract_tags(threat_data),
            evidence=[threat_data],
            metadata=threat_assessment
        )
        
        # Add to incident management
        self.incidents[incident_id] = incident
        self.incident_queue.append(incident)
        self.active_incidents.add(incident_id)
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.now(),
            "event": "incident_detected",
            "description": "Incident automatically detected",
            "source": "system"
        })
        
        self.logger.info(f"Incident detected: {incident_id} - {incident.title}")
        
        # Trigger automated response if enabled
        if self.auto_response:
            asyncio.create_task(self._trigger_automated_response(incident))
        
        return incident
    
    def _should_create_incident(self, threat_data: Dict[str, Any], 
                               threat_assessment: Dict[str, Any]) -> bool:
        """Determine if an incident should be created."""
        # Check severity threshold
        risk_level = threat_assessment.get("risk_level", "low")
        severity_thresholds = {
            "low": False,  # Don't create incidents for low risk
            "medium": True,
            "high": True,
            "critical": True
        }
        
        if not severity_thresholds.get(risk_level, False):
            return False
        
        # Check for duplicate incidents
        if self._is_duplicate_incident(threat_data):
            return False
        
        # Check confidence threshold
        confidence = threat_assessment.get("confidence", 0.0)
        if confidence < 0.6:  # Minimum confidence threshold
            return False
        
        return True
    
    def _is_duplicate_incident(self, threat_data: Dict[str, Any]) -> bool:
        """Check if this is a duplicate of an existing incident."""
        # Simple duplicate detection based on key attributes
        threat_signature = self._generate_threat_signature(threat_data)
        
        for incident in self.incidents.values():
            if incident.status in [IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE]:
                continue
            
            incident_signature = self._generate_threat_signature(incident.evidence[0] if incident.evidence else {})
            if threat_signature == incident_signature:
                return True
        
        return False
    
    def _generate_threat_signature(self, threat_data: Dict[str, Any]) -> str:
        """Generate a signature for threat data to detect duplicates."""
        # Create a signature based on key identifying attributes
        signature_parts = []
        
        if "source_ip" in threat_data:
            signature_parts.append(f"ip:{threat_data['source_ip']}")
        if "file_hash" in threat_data:
            signature_parts.append(f"hash:{threat_data['file_hash']}")
        if "user_id" in threat_data:
            signature_parts.append(f"user:{threat_data['user_id']}")
        if "process_name" in threat_data:
            signature_parts.append(f"process:{threat_data['process_name']}")
        
        return "|".join(signature_parts)
    
    def _classify_severity(self, threat_assessment: Dict[str, Any]) -> IncidentSeverity:
        """Classify incident severity."""
        risk_level = threat_assessment.get("risk_level", "low")
        
        severity_mapping = {
            "low": IncidentSeverity.LOW,
            "medium": IncidentSeverity.MEDIUM,
            "high": IncidentSeverity.HIGH,
            "critical": IncidentSeverity.CRITICAL
        }
        
        return severity_mapping.get(risk_level, IncidentSeverity.LOW)
    
    def _classify_category(self, threat_data: Dict[str, Any]) -> str:
        """Classify incident category."""
        # Simple category classification based on threat data
        if threat_data.get("malware_detected"):
            return "malware"
        elif threat_data.get("network_intrusion"):
            return "network_intrusion"
        elif threat_data.get("data_breach"):
            return "data_breach"
        elif threat_data.get("dos_attack"):
            return "dos_attack"
        elif threat_data.get("phishing"):
            return "phishing"
        else:
            return "unknown"
    
    def _generate_incident_title(self, threat_data: Dict[str, Any], category: str) -> str:
        """Generate incident title."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        source = threat_data.get("source", "unknown")
        
        return f"{category.replace('_', ' ').title()} - {source} - {timestamp}"
    
    def _generate_incident_description(self, threat_data: Dict[str, Any], 
                                     threat_assessment: Dict[str, Any]) -> str:
        """Generate incident description."""
        description_parts = []
        
        # Add threat type
        threat_type = threat_data.get("threat_type", "unknown")
        description_parts.append(f"Threat Type: {threat_type}")
        
        # Add source information
        if "source_ip" in threat_data:
            description_parts.append(f"Source IP: {threat_data['source_ip']}")
        if "source_user" in threat_data:
            description_parts.append(f"Source User: {threat_data['source_user']}")
        
        # Add confidence and risk level
        confidence = threat_assessment.get("confidence", 0.0)
        risk_level = threat_assessment.get("risk_level", "unknown")
        description_parts.append(f"Risk Level: {risk_level} (confidence: {confidence:.2f})")
        
        # Add key indicators
        if "indicators" in threat_data:
            description_parts.append(f"Indicators: {', '.join(threat_data['indicators'])}")
        
        return "\n".join(description_parts)
    
    def _extract_tags(self, threat_data: Dict[str, Any]) -> Set[str]:
        """Extract tags from threat data."""
        tags = set()
        
        # Add category-based tags
        if threat_data.get("malware_detected"):
            tags.add("malware")
        if threat_data.get("network_intrusion"):
            tags.add("network")
        if threat_data.get("data_breach"):
            tags.add("data_breach")
        
        # Add severity-based tags
        if threat_data.get("critical"):
            tags.add("critical")
        if threat_data.get("immediate_threat"):
            tags.add("immediate")
        
        # Add source-based tags
        if threat_data.get("external_source"):
            tags.add("external")
        if threat_data.get("internal_source"):
            tags.add("internal")
        
        return tags
    
    async def _trigger_automated_response(self, incident: Incident):
        """Trigger automated response workflows for an incident."""
        try:
            # Find applicable workflows
            applicable_workflows = self._find_applicable_workflows(incident)
            
            for workflow in applicable_workflows:
                await self._execute_workflow(workflow, incident)
            
            # Update incident status
            if incident.status == IncidentStatus.DETECTED:
                incident.status = IncidentStatus.INVESTIGATING
                incident.timeline.append({
                    "timestamp": datetime.now(),
                    "event": "status_change",
                    "description": "Status changed to investigating",
                    "source": "automated_response"
                })
            
        except Exception as e:
            self.logger.error(f"Error in automated response: {e}")
            incident.timeline.append({
                "timestamp": datetime.now(),
                "event": "error",
                "description": f"Automated response error: {str(e)}",
                "source": "system"
            })
    
    def _find_applicable_workflows(self, incident: Incident) -> List[ResponseWorkflow]:
        """Find workflows applicable to an incident."""
        applicable_workflows = []
        
        for workflow in self.response_workflows.values():
            if not workflow.enabled:
                continue
            
            if self._workflow_matches_incident(workflow, incident):
                applicable_workflows.append(workflow)
        
        # Sort by priority (lower number = higher priority)
        applicable_workflows.sort(key=lambda w: w.priority)
        
        return applicable_workflows
    
    def _workflow_matches_incident(self, workflow: ResponseWorkflow, incident: Incident) -> bool:
        """Check if a workflow matches an incident."""
        for trigger in workflow.triggers:
            trigger_type = trigger["type"]
            trigger_value = trigger["value"]
            
            if trigger_type == "threat_category":
                if incident.category != trigger_value:
                    return False
            elif trigger_type == "severity":
                if incident.severity.value != trigger_value:
                    return False
            elif trigger_type == "source":
                if incident.source != trigger_value:
                    return False
            # Add more trigger types as needed
        
        # Check conditions
        for condition in workflow.conditions:
            if not self._evaluate_condition(condition, incident):
                return False
        
        return True
    
    def _evaluate_condition(self, condition: Dict[str, Any], incident: Incident) -> bool:
        """Evaluate a workflow condition against an incident."""
        condition_type = condition["type"]
        condition_value = condition["value"]
        operator = condition.get("operator", "==")
        
        # Get the value to compare
        if condition_type == "file_type":
            # Check evidence for file type
            for evidence in incident.evidence:
                if evidence.get("file_type") == condition_value:
                    return True
            return False
        elif condition_type == "confidence":
            confidence = incident.metadata.get("confidence", 0.0)
            if operator == ">":
                return confidence > condition_value
            elif operator == ">=":
                return confidence >= condition_value
            elif operator == "<":
                return confidence < condition_value
            elif operator == "<=":
                return confidence <= condition_value
            else:
                return confidence == condition_value
        elif condition_type == "network_activity":
            # Check evidence for network activity
            for evidence in incident.evidence:
                if evidence.get("network_activity") == condition_value:
                    return True
            return False
        elif condition_type == "data_sensitivity":
            # Check evidence for data sensitivity
            for evidence in incident.evidence:
                if evidence.get("data_sensitivity") == condition_value:
                    return True
            return False
        
        return True  # Default to true for unknown conditions
    
    async def _execute_workflow(self, workflow: ResponseWorkflow, incident: Incident):
        """Execute a response workflow."""
        self.logger.info(f"Executing workflow: {workflow.name} for incident: {incident.incident_id}")
        
        for action in workflow.actions:
            try:
                await self._execute_response_action(action, incident)
                
                # Record action in incident
                incident.response_actions.append({
                    "timestamp": datetime.now(),
                    "action": action.value,
                    "workflow": workflow.workflow_id,
                    "status": "completed"
                })
                
                # Add to timeline
                incident.timeline.append({
                    "timestamp": datetime.now(),
                    "event": "response_action",
                    "description": f"Executed {action.value}",
                    "source": "automated_response"
                })
                
            except Exception as e:
                self.logger.error(f"Error executing action {action.value}: {e}")
                
                # Record failed action
                incident.response_actions.append({
                    "timestamp": datetime.now(),
                    "action": action.value,
                    "workflow": workflow.workflow_id,
                    "status": "failed",
                    "error": str(e)
                })
    
    async def _execute_response_action(self, action: ResponseAction, incident: Incident):
        """Execute a specific response action."""
        if action == ResponseAction.ISOLATE_SYSTEM:
            await self._isolate_system(incident)
        elif action == ResponseAction.BLOCK_IP:
            await self._block_ip(incident)
        elif action == ResponseAction.QUARANTINE_FILE:
            await self._quarantine_file(incident)
        elif action == ResponseAction.RESET_CREDENTIALS:
            await self._reset_credentials(incident)
        elif action == ResponseAction.NOTIFY_TEAM:
            await self._notify_team(incident)
        elif action == ResponseAction.ESCALATE:
            await self._escalate_incident(incident)
        elif action == ResponseAction.MONITOR:
            await self._monitor_incident(incident)
        elif action == ResponseAction.COLLECT_EVIDENCE:
            await self._collect_evidence(incident)
        elif action == ResponseAction.UPDATE_SIGNATURES:
            await self._update_signatures(incident)
        elif action == ResponseAction.PATCH_SYSTEM:
            await self._patch_system(incident)
    
    async def _isolate_system(self, incident: Incident):
        """Isolate affected system."""
        self.logger.info(f"Isolating system for incident: {incident.incident_id}")
        # Implementation would depend on specific infrastructure
        # This is a placeholder for the actual isolation logic
        await asyncio.sleep(0.1)  # Simulate async operation
    
    async def _block_ip(self, incident: Incident):
        """Block suspicious IP address."""
        self.logger.info(f"Blocking IP for incident: {incident.incident_id}")
        # Implementation would depend on firewall/network infrastructure
        await asyncio.sleep(0.1)
    
    async def _quarantine_file(self, incident: Incident):
        """Quarantine suspicious file."""
        self.logger.info(f"Quarantining file for incident: {incident.incident_id}")
        # Implementation would depend on file system and security tools
        await asyncio.sleep(0.1)
    
    async def _reset_credentials(self, incident: Incident):
        """Reset compromised credentials."""
        self.logger.info(f"Resetting credentials for incident: {incident.incident_id}")
        # Implementation would depend on identity management system
        await asyncio.sleep(0.1)
    
    async def _notify_team(self, incident: Incident):
        """Notify response team."""
        self.logger.info(f"Notifying team for incident: {incident.incident_id}")
        # Implementation would depend on notification system
        await asyncio.sleep(0.1)
    
    async def _escalate_incident(self, incident: Incident):
        """Escalate incident to higher priority."""
        self.logger.info(f"Escalating incident: {incident.incident_id}")
        # Implementation would depend on escalation procedures
        await asyncio.sleep(0.1)
    
    async def _monitor_incident(self, incident: Incident):
        """Monitor incident for changes."""
        self.logger.info(f"Monitoring incident: {incident.incident_id}")
        # Implementation would depend on monitoring system
        await asyncio.sleep(0.1)
    
    async def _collect_evidence(self, incident: Incident):
        """Collect evidence for incident."""
        self.logger.info(f"Collecting evidence for incident: {incident.incident_id}")
        # Implementation would depend on evidence collection tools
        await asyncio.sleep(0.1)
    
    async def _update_signatures(self, incident: Incident):
        """Update security signatures."""
        self.logger.info(f"Updating signatures for incident: {incident.incident_id}")
        # Implementation would depend on signature management system
        await asyncio.sleep(0.1)
    
    async def _patch_system(self, incident: Incident):
        """Apply security patches."""
        self.logger.info(f"Patching system for incident: {incident.incident_id}")
        # Implementation would depend on patch management system
        await asyncio.sleep(0.1)
    
    def update_incident_status(self, incident_id: str, new_status: IncidentStatus, 
                              analyst: Optional[str] = None, notes: Optional[str] = None):
        """Update incident status."""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        old_status = incident.status
        incident.status = new_status
        
        # Update assigned analyst if provided
        if analyst:
            incident.assigned_analyst = analyst
        
        # Add to timeline
        timeline_entry = {
            "timestamp": datetime.now(),
            "event": "status_change",
            "description": f"Status changed from {old_status.value} to {new_status.value}",
            "source": "manual" if analyst else "system"
        }
        
        if analyst:
            timeline_entry["analyst"] = analyst
        if notes:
            timeline_entry["notes"] = notes
        
        incident.timeline.append(timeline_entry)
        
        # Update active incidents
        if new_status in [IncidentStatus.CLOSED, IncidentStatus.FALSE_POSITIVE]:
            self.active_incidents.discard(incident_id)
        else:
            self.active_incidents.add(incident_id)
        
        # Set resolution time if closed
        if new_status == IncidentStatus.CLOSED:
            incident.resolution_time = datetime.now()
        
        self.logger.info(f"Incident {incident_id} status updated to {new_status.value}")
    
    def add_evidence(self, incident_id: str, evidence: Dict[str, Any]):
        """Add evidence to an incident."""
        if incident_id not in self.incidents:
            raise ValueError(f"Incident {incident_id} not found")
        
        incident = self.incidents[incident_id]
        incident.evidence.append(evidence)
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": datetime.now(),
            "event": "evidence_added",
            "description": f"Evidence added: {evidence.get('type', 'unknown')}",
            "source": "manual"
        })
    
    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get incident by ID."""
        return self.incidents.get(incident_id)
    
    def get_active_incidents(self) -> List[Incident]:
        """Get all active incidents."""
        return [self.incidents[incident_id] for incident_id in self.active_incidents 
                if incident_id in self.incidents]
    
    def get_incidents_by_severity(self, severity: IncidentSeverity) -> List[Incident]:
        """Get incidents by severity."""
        return [incident for incident in self.incidents.values() 
                if incident.severity == severity]
    
    def get_incidents_by_status(self, status: IncidentStatus) -> List[Incident]:
        """Get incidents by status."""
        return [incident for incident in self.incidents.values() 
                if incident.status == status]
    
    def get_incident_metrics(self) -> Dict[str, Any]:
        """Get incident response metrics."""
        total_incidents = len(self.incidents)
        active_incidents = len(self.active_incidents)
        
        # Calculate status distribution
        status_distribution = defaultdict(int)
        for incident in self.incidents.values():
            status_distribution[incident.status.value] += 1
        
        # Calculate severity distribution
        severity_distribution = defaultdict(int)
        for incident in self.incidents.values():
            severity_distribution[incident.severity.value] += 1
        
        # Calculate average resolution time
        resolved_incidents = [i for i in self.incidents.values() 
                            if i.resolution_time and i.detection_time]
        avg_resolution_time = 0.0
        if resolved_incidents:
            total_time = sum((i.resolution_time - i.detection_time).total_seconds() 
                           for i in resolved_incidents)
            avg_resolution_time = total_time / len(resolved_incidents)
        
        # Calculate response time metrics
        response_times = []
        for incident in self.incidents.values():
            if incident.timeline:
                detection_time = incident.detection_time
                first_response = None
                for event in incident.timeline:
                    if event["event"] == "response_action":
                        first_response = event["timestamp"]
                        break
                
                if first_response:
                    response_time = (first_response - detection_time).total_seconds()
                    response_times.append(response_time)
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        return {
            "total_incidents": total_incidents,
            "active_incidents": active_incidents,
            "status_distribution": dict(status_distribution),
            "severity_distribution": dict(severity_distribution),
            "average_resolution_time_seconds": avg_resolution_time,
            "average_response_time_seconds": avg_response_time,
            "auto_response_enabled": self.auto_response,
            "total_workflows": len(self.response_workflows)
        }
