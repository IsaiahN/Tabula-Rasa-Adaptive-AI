#!/usr/bin/env python3
"""
Security Audit System - Comprehensive security auditing and compliance.

This module provides:
- Security policy compliance checking
- Vulnerability assessment
- Security configuration auditing
- Compliance reporting
- Security posture analysis
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
import hashlib

logger = logging.getLogger(__name__)


class ComplianceStandard(Enum):
    """Compliance standards."""
    NIST = "nist"
    ISO27001 = "iso27001"
    SOC2 = "soc2"
    PCI_DSS = "pci_dss"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CUSTOM = "custom"


class AuditSeverity(Enum):
    """Audit finding severity levels."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuditStatus(Enum):
    """Audit status states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SecurityPolicy:
    """Security policy definition."""
    policy_id: str
    name: str
    description: str
    category: str
    requirements: List[Dict[str, Any]]
    compliance_standard: ComplianceStandard
    severity: AuditSeverity
    enabled: bool = True
    last_updated: Optional[datetime] = None


@dataclass
class AuditFinding:
    """Security audit finding."""
    finding_id: str
    policy_id: str
    title: str
    description: str
    severity: AuditSeverity
    status: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    remediation_steps: List[str] = field(default_factory=list)
    discovered_time: datetime = field(default_factory=datetime.now)
    resolved_time: Optional[datetime] = None
    assigned_to: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditReport:
    """Security audit report."""
    report_id: str
    audit_name: str
    audit_type: str
    compliance_standard: ComplianceStandard
    start_time: datetime
    end_time: Optional[datetime] = None
    status: AuditStatus = AuditStatus.PENDING
    findings: List[AuditFinding] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityAuditSystem:
    """
    Comprehensive security audit system.
    
    Provides:
    - Security policy compliance checking
    - Vulnerability assessment
    - Security configuration auditing
    - Compliance reporting
    - Security posture analysis
    """
    
    def __init__(self, max_findings: int = 10000, max_reports: int = 1000):
        self.max_findings = max_findings
        self.max_reports = max_reports
        self.logger = logging.getLogger(__name__)
        
        # Security policies
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.policy_categories: Dict[str, List[str]] = defaultdict(list)
        
        # Audit findings
        self.audit_findings: Dict[str, AuditFinding] = {}
        self.findings_by_policy: Dict[str, List[str]] = defaultdict(list)
        self.findings_by_severity: Dict[AuditSeverity, List[str]] = defaultdict(list)
        
        # Audit reports
        self.audit_reports: Dict[str, AuditReport] = {}
        self.active_audits: Set[str] = set()
        
        # Compliance tracking
        self.compliance_scores: Dict[ComplianceStandard, float] = {}
        self.compliance_history: Dict[ComplianceStandard, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Initialize default policies
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default security policies."""
        # Password policy
        password_policy = SecurityPolicy(
            policy_id="password_policy",
            name="Password Security Policy",
            description="Requirements for password security",
            category="authentication",
            requirements=[
                {
                    "id": "min_length",
                    "description": "Minimum password length",
                    "type": "numeric",
                    "operator": ">=",
                    "value": 8,
                    "severity": AuditSeverity.HIGH
                },
                {
                    "id": "complexity",
                    "description": "Password complexity requirements",
                    "type": "regex",
                    "pattern": r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]",
                    "severity": AuditSeverity.HIGH
                },
                {
                    "id": "history",
                    "description": "Password history enforcement",
                    "type": "numeric",
                    "operator": ">=",
                    "value": 5,
                    "severity": AuditSeverity.MEDIUM
                }
            ],
            compliance_standard=ComplianceStandard.NIST,
            severity=AuditSeverity.HIGH
        )
        self._add_policy(password_policy)
        
        # Access control policy
        access_policy = SecurityPolicy(
            policy_id="access_control",
            name="Access Control Policy",
            description="Requirements for access control",
            category="access_control",
            requirements=[
                {
                    "id": "principle_of_least_privilege",
                    "description": "Principle of least privilege",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.CRITICAL
                },
                {
                    "id": "regular_access_review",
                    "description": "Regular access review",
                    "type": "frequency",
                    "value": "quarterly",
                    "severity": AuditSeverity.HIGH
                },
                {
                    "id": "multi_factor_auth",
                    "description": "Multi-factor authentication",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.HIGH
                }
            ],
            compliance_standard=ComplianceStandard.NIST,
            severity=AuditSeverity.HIGH
        )
        self._add_policy(access_policy)
        
        # Data encryption policy
        encryption_policy = SecurityPolicy(
            policy_id="data_encryption",
            name="Data Encryption Policy",
            description="Requirements for data encryption",
            category="data_protection",
            requirements=[
                {
                    "id": "encryption_at_rest",
                    "description": "Encryption at rest",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.CRITICAL
                },
                {
                    "id": "encryption_in_transit",
                    "description": "Encryption in transit",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.CRITICAL
                },
                {
                    "id": "encryption_algorithm",
                    "description": "Strong encryption algorithm",
                    "type": "enum",
                    "values": ["AES-256", "ChaCha20-Poly1305"],
                    "severity": AuditSeverity.HIGH
                }
            ],
            compliance_standard=ComplianceStandard.NIST,
            severity=AuditSeverity.CRITICAL
        )
        self._add_policy(encryption_policy)
        
        # Network security policy
        network_policy = SecurityPolicy(
            policy_id="network_security",
            name="Network Security Policy",
            description="Requirements for network security",
            category="network_security",
            requirements=[
                {
                    "id": "firewall_enabled",
                    "description": "Firewall enabled",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.HIGH
                },
                {
                    "id": "intrusion_detection",
                    "description": "Intrusion detection system",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.HIGH
                },
                {
                    "id": "network_segmentation",
                    "description": "Network segmentation",
                    "type": "boolean",
                    "value": True,
                    "severity": AuditSeverity.MEDIUM
                }
            ],
            compliance_standard=ComplianceStandard.NIST,
            severity=AuditSeverity.HIGH
        )
        self._add_policy(network_policy)
    
    def _add_policy(self, policy: SecurityPolicy):
        """Add a security policy."""
        self.security_policies[policy.policy_id] = policy
        self.policy_categories[policy.category].append(policy.policy_id)
        self.logger.info(f"Added security policy: {policy.name}")
    
    def create_audit(self, audit_name: str, audit_type: str, 
                    compliance_standard: ComplianceStandard,
                    policies: Optional[List[str]] = None) -> str:
        """
        Create a new security audit.
        
        Args:
            audit_name: Name of the audit
            audit_type: Type of audit (e.g., "compliance", "vulnerability")
            compliance_standard: Compliance standard to audit against
            policies: List of policy IDs to audit (None for all)
            
        Returns:
            Audit report ID
        """
        report_id = str(uuid.uuid4())
        
        # Determine policies to audit
        if policies is None:
            policies = list(self.security_policies.keys())
        
        # Create audit report
        report = AuditReport(
            report_id=report_id,
            audit_name=audit_name,
            audit_type=audit_type,
            compliance_standard=compliance_standard,
            start_time=datetime.now(),
            status=AuditStatus.IN_PROGRESS
        )
        
        self.audit_reports[report_id] = report
        self.active_audits.add(report_id)
        
        self.logger.info(f"Created audit: {audit_name} ({report_id})")
        
        return report_id
    
    def run_audit(self, report_id: str, system_data: Dict[str, Any]) -> AuditReport:
        """
        Run a security audit.
        
        Args:
            report_id: ID of the audit report
            system_data: System configuration and data to audit
            
        Returns:
            Updated audit report
        """
        if report_id not in self.audit_reports:
            raise ValueError(f"Audit report {report_id} not found")
        
        report = self.audit_reports[report_id]
        report.status = AuditStatus.IN_PROGRESS
        
        self.logger.info(f"Running audit: {report.audit_name}")
        
        try:
            # Run policy compliance checks
            findings = []
            for policy_id in self._get_audit_policies(report):
                policy = self.security_policies[policy_id]
                policy_findings = self._audit_policy_compliance(policy, system_data)
                findings.extend(policy_findings)
            
            # Add findings to report
            report.findings = findings
            
            # Generate summary
            report.summary = self._generate_audit_summary(findings)
            
            # Generate recommendations
            report.recommendations = self._generate_audit_recommendations(findings)
            
            # Update compliance scores
            self._update_compliance_scores(report)
            
            # Mark as completed
            report.status = AuditStatus.COMPLETED
            report.end_time = datetime.now()
            
            # Remove from active audits
            self.active_audits.discard(report_id)
            
            self.logger.info(f"Audit completed: {report.audit_name}")
            
        except Exception as e:
            self.logger.error(f"Audit failed: {e}")
            report.status = AuditStatus.FAILED
            report.metadata["error"] = str(e)
        
        return report
    
    def _get_audit_policies(self, report: AuditReport) -> List[str]:
        """Get policies to audit for a report."""
        # Filter policies by compliance standard
        relevant_policies = []
        for policy in self.security_policies.values():
            if policy.compliance_standard == report.compliance_standard:
                relevant_policies.append(policy.policy_id)
        
        return relevant_policies
    
    def _audit_policy_compliance(self, policy: SecurityPolicy, 
                               system_data: Dict[str, Any]) -> List[AuditFinding]:
        """Audit compliance with a specific policy."""
        findings = []
        
        for requirement in policy.requirements:
            finding = self._check_requirement_compliance(
                policy, requirement, system_data
            )
            if finding:
                findings.append(finding)
        
        return findings
    
    def _check_requirement_compliance(self, policy: SecurityPolicy, 
                                    requirement: Dict[str, Any], 
                                    system_data: Dict[str, Any]) -> Optional[AuditFinding]:
        """Check compliance with a specific requirement."""
        req_id = requirement["id"]
        req_type = requirement["type"]
        req_value = requirement["value"]
        req_severity = requirement.get("severity", AuditSeverity.MEDIUM)
        
        # Get system value for this requirement
        system_value = self._get_system_value(system_data, req_id)
        
        # Check compliance based on requirement type
        is_compliant = self._evaluate_requirement(req_type, req_value, system_value, requirement)
        
        if is_compliant:
            return None  # No finding if compliant
        
        # Create finding for non-compliance
        finding_id = str(uuid.uuid4())
        
        finding = AuditFinding(
            finding_id=finding_id,
            policy_id=policy.policy_id,
            title=f"Non-compliance: {requirement['description']}",
            description=f"Requirement '{requirement['description']}' is not met. "
                       f"Expected: {req_value}, Found: {system_value}",
            severity=req_severity,
            status="open",
            evidence=[{
                "type": "requirement_check",
                "requirement_id": req_id,
                "expected_value": req_value,
                "actual_value": system_value,
                "timestamp": datetime.now()
            }],
            recommendations=self._generate_requirement_recommendations(requirement),
            remediation_steps=self._generate_remediation_steps(requirement),
            tags={policy.category, policy.compliance_standard.value}
        )
        
        # Store finding
        self.audit_findings[finding_id] = finding
        self.findings_by_policy[policy.policy_id].append(finding_id)
        self.findings_by_severity[req_severity].append(finding_id)
        
        return finding
    
    def _get_system_value(self, system_data: Dict[str, Any], req_id: str) -> Any:
        """Get system value for a requirement ID."""
        # This is a simplified implementation
        # In practice, this would map requirement IDs to actual system data
        
        # Common mappings
        value_mappings = {
            "min_length": system_data.get("password_min_length", 0),
            "complexity": system_data.get("password_complexity_enabled", False),
            "history": system_data.get("password_history_count", 0),
            "principle_of_least_privilege": system_data.get("least_privilege_enabled", False),
            "regular_access_review": system_data.get("access_review_frequency", "never"),
            "multi_factor_auth": system_data.get("mfa_enabled", False),
            "encryption_at_rest": system_data.get("encryption_at_rest_enabled", False),
            "encryption_in_transit": system_data.get("encryption_in_transit_enabled", False),
            "encryption_algorithm": system_data.get("encryption_algorithm", "none"),
            "firewall_enabled": system_data.get("firewall_enabled", False),
            "intrusion_detection": system_data.get("ids_enabled", False),
            "network_segmentation": system_data.get("network_segmentation_enabled", False)
        }
        
        return value_mappings.get(req_id, None)
    
    def _evaluate_requirement(self, req_type: str, req_value: Any, 
                            system_value: Any, requirement: Dict[str, Any]) -> bool:
        """Evaluate if a requirement is met."""
        if system_value is None:
            return False  # No system value means non-compliant
        
        if req_type == "boolean":
            return bool(system_value) == bool(req_value)
        elif req_type == "numeric":
            operator = requirement.get("operator", "==")
            if operator == ">=":
                return system_value >= req_value
            elif operator == ">":
                return system_value > req_value
            elif operator == "<=":
                return system_value <= req_value
            elif operator == "<":
                return system_value < req_value
            else:
                return system_value == req_value
        elif req_type == "string":
            return str(system_value) == str(req_value)
        elif req_type == "regex":
            import re
            pattern = requirement.get("pattern", "")
            return bool(re.match(pattern, str(system_value)))
        elif req_type == "enum":
            allowed_values = requirement.get("values", [])
            return system_value in allowed_values
        elif req_type == "frequency":
            # Convert frequency to comparable values
            frequency_values = {
                "daily": 1,
                "weekly": 7,
                "monthly": 30,
                "quarterly": 90,
                "yearly": 365,
                "never": float('inf')
            }
            system_freq = frequency_values.get(system_value, float('inf'))
            required_freq = frequency_values.get(req_value, float('inf'))
            return system_freq <= required_freq
        else:
            return False  # Unknown requirement type
    
    def _generate_requirement_recommendations(self, requirement: Dict[str, Any]) -> List[str]:
        """Generate recommendations for a requirement."""
        req_id = requirement["id"]
        req_type = requirement["type"]
        req_value = requirement["value"]
        
        recommendations = []
        
        if req_id == "min_length":
            recommendations.append(f"Set minimum password length to at least {req_value} characters")
        elif req_id == "complexity":
            recommendations.append("Enable password complexity requirements")
        elif req_id == "history":
            recommendations.append(f"Implement password history to prevent reuse of last {req_value} passwords")
        elif req_id == "principle_of_least_privilege":
            recommendations.append("Implement principle of least privilege for all user accounts")
        elif req_id == "regular_access_review":
            recommendations.append(f"Implement regular access reviews on a {req_value} basis")
        elif req_id == "multi_factor_auth":
            recommendations.append("Enable multi-factor authentication for all user accounts")
        elif req_id == "encryption_at_rest":
            recommendations.append("Enable encryption for all data at rest")
        elif req_id == "encryption_in_transit":
            recommendations.append("Enable encryption for all data in transit")
        elif req_id == "encryption_algorithm":
            recommendations.append(f"Use strong encryption algorithm: {req_value}")
        elif req_id == "firewall_enabled":
            recommendations.append("Enable and properly configure firewall")
        elif req_id == "intrusion_detection":
            recommendations.append("Implement intrusion detection system")
        elif req_id == "network_segmentation":
            recommendations.append("Implement network segmentation")
        
        return recommendations
    
    def _generate_remediation_steps(self, requirement: Dict[str, Any]) -> List[str]:
        """Generate remediation steps for a requirement."""
        req_id = requirement["id"]
        
        steps = []
        
        if req_id == "min_length":
            steps.extend([
                "Access system configuration",
                "Navigate to password policy settings",
                "Set minimum password length",
                "Apply changes",
                "Test with new password"
            ])
        elif req_id == "complexity":
            steps.extend([
                "Access system configuration",
                "Navigate to password policy settings",
                "Enable complexity requirements",
                "Configure complexity rules",
                "Apply changes"
            ])
        elif req_id == "multi_factor_auth":
            steps.extend([
                "Choose MFA solution",
                "Configure MFA server",
                "Enroll users in MFA",
                "Test MFA functionality",
                "Update user documentation"
            ])
        elif req_id == "encryption_at_rest":
            steps.extend([
                "Identify data storage locations",
                "Choose encryption solution",
                "Implement encryption",
                "Test encryption functionality",
                "Update data handling procedures"
            ])
        else:
            steps.extend([
                "Review current configuration",
                "Identify gaps",
                "Implement required changes",
                "Test functionality",
                "Document changes"
            ])
        
        return steps
    
    def _generate_audit_summary(self, findings: List[AuditFinding]) -> Dict[str, Any]:
        """Generate audit summary."""
        total_findings = len(findings)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for finding in findings:
            severity_counts[finding.severity.value] += 1
        
        # Count by policy
        policy_counts = defaultdict(int)
        for finding in findings:
            policy_counts[finding.policy_id] += 1
        
        # Calculate compliance score
        total_requirements = sum(len(policy.requirements) for policy in self.security_policies.values())
        compliant_requirements = total_requirements - total_findings
        compliance_score = (compliant_requirements / total_requirements * 100) if total_requirements > 0 else 100
        
        return {
            "total_findings": total_findings,
            "severity_distribution": dict(severity_counts),
            "policy_distribution": dict(policy_counts),
            "compliance_score": compliance_score,
            "critical_findings": severity_counts[AuditSeverity.CRITICAL.value],
            "high_findings": severity_counts[AuditSeverity.HIGH.value],
            "medium_findings": severity_counts[AuditSeverity.MEDIUM.value],
            "low_findings": severity_counts[AuditSeverity.LOW.value]
        }
    
    def _generate_audit_recommendations(self, findings: List[AuditFinding]) -> List[str]:
        """Generate audit recommendations."""
        recommendations = []
        
        # Group findings by severity
        critical_findings = [f for f in findings if f.severity == AuditSeverity.CRITICAL]
        high_findings = [f for f in findings if f.severity == AuditSeverity.HIGH]
        
        if critical_findings:
            recommendations.append("Immediately address all critical findings")
            recommendations.append("Implement emergency response procedures")
        
        if high_findings:
            recommendations.append("Prioritize high-severity findings for immediate remediation")
        
        # Add specific recommendations
        for finding in findings[:10]:  # Limit to top 10 findings
            recommendations.extend(finding.recommendations[:2])  # Top 2 per finding
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:20]  # Limit to 20 recommendations
    
    def _update_compliance_scores(self, report: AuditReport):
        """Update compliance scores based on audit results."""
        if report.status != AuditStatus.COMPLETED:
            return
        
        compliance_standard = report.compliance_standard
        summary = report.summary
        
        if summary:
            compliance_score = summary.get("compliance_score", 0.0)
            self.compliance_scores[compliance_standard] = compliance_score
            
            # Add to history
            self.compliance_history[compliance_standard].append(
                (report.end_time, compliance_score)
            )
            
            # Keep only recent history (last 100 entries)
            if len(self.compliance_history[compliance_standard]) > 100:
                self.compliance_history[compliance_standard] = \
                    self.compliance_history[compliance_standard][-100:]
    
    def get_audit_report(self, report_id: str) -> Optional[AuditReport]:
        """Get audit report by ID."""
        return self.audit_reports.get(report_id)
    
    def get_findings_by_severity(self, severity: AuditSeverity) -> List[AuditFinding]:
        """Get findings by severity."""
        finding_ids = self.findings_by_severity.get(severity, [])
        return [self.audit_findings[fid] for fid in finding_ids if fid in self.audit_findings]
    
    def get_findings_by_policy(self, policy_id: str) -> List[AuditFinding]:
        """Get findings by policy."""
        finding_ids = self.findings_by_policy.get(policy_id, [])
        return [self.audit_findings[fid] for fid in finding_ids if fid in self.audit_findings]
    
    def get_compliance_score(self, standard: ComplianceStandard) -> float:
        """Get current compliance score for a standard."""
        return self.compliance_scores.get(standard, 0.0)
    
    def get_compliance_trends(self, standard: ComplianceStandard, 
                            days: int = 30) -> List[Tuple[datetime, float]]:
        """Get compliance trends for a standard."""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [(date, score) for date, score in self.compliance_history[standard] 
                if date >= cutoff_date]
    
    def add_security_policy(self, policy: SecurityPolicy):
        """Add a new security policy."""
        self._add_policy(policy)
    
    def update_finding_status(self, finding_id: str, status: str, 
                            assigned_to: Optional[str] = None, notes: Optional[str] = None):
        """Update finding status."""
        if finding_id not in self.audit_findings:
            raise ValueError(f"Finding {finding_id} not found")
        
        finding = self.audit_findings[finding_id]
        finding.status = status
        
        if assigned_to:
            finding.assigned_to = assigned_to
        
        if status == "resolved":
            finding.resolved_time = datetime.now()
        
        if notes:
            finding.metadata["notes"] = notes
        
        self.logger.info(f"Updated finding {finding_id} status to {status}")
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """Get audit system metrics."""
        total_policies = len(self.security_policies)
        total_findings = len(self.audit_findings)
        total_reports = len(self.audit_reports)
        active_audits = len(self.active_audits)
        
        # Calculate finding distribution by severity
        severity_distribution = {}
        for severity in AuditSeverity:
            count = len(self.findings_by_severity[severity])
            severity_distribution[severity.value] = count
        
        # Calculate policy distribution by category
        category_distribution = {}
        for category, policy_ids in self.policy_categories.items():
            category_distribution[category] = len(policy_ids)
        
        # Calculate average compliance scores
        avg_compliance = {}
        for standard, score in self.compliance_scores.items():
            avg_compliance[standard.value] = score
        
        return {
            "total_policies": total_policies,
            "total_findings": total_findings,
            "total_reports": total_reports,
            "active_audits": active_audits,
            "severity_distribution": severity_distribution,
            "category_distribution": category_distribution,
            "compliance_scores": avg_compliance,
            "open_findings": len([f for f in self.audit_findings.values() if f.status == "open"]),
            "resolved_findings": len([f for f in self.audit_findings.values() if f.status == "resolved"])
        }
