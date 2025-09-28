"""Agentic capabilities for external API interactions"""

import requests
from typing import Dict, Any, Optional, List
import json


class AgenticHandler:
    """Handles agentic tasks like API requests and data fetching"""
    
    def __init__(self):
        self.session = requests.Session()
        self.supported_actions = [
            "fetch_user_profile",
            "fetch_academic_records",
            "fetch_attendance",
            "fetch_performance_data",
            "submit_request",
            "check_request_status"
        ]
    
    def authenticate_user(self, token: str, organization_api: str) -> Dict[str, Any]:
        """Authenticate user with organization's API"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{organization_api}/auth/verify",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {
                    "success": True,
                    "user_data": response.json()
                }
            else:
                return {
                    "success": False,
                    "error": f"Authentication failed: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Authentication error: {str(e)}"
            }
    
    def fetch_user_profile(self, token: str, organization_api: str, user_id: str) -> Dict[str, Any]:
        """Fetch user profile data"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{organization_api}/users/{user_id}/profile",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                profile_data = response.json()
                return {
                    "success": True,
                    "data": profile_data,
                    "summary": self._generate_profile_summary(profile_data)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to fetch profile: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Profile fetch error: {str(e)}"
            }
    
    def fetch_academic_records(self, token: str, organization_api: str, user_id: str) -> Dict[str, Any]:
        """Fetch academic records for students"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            response = self.session.get(
                f"{organization_api}/students/{user_id}/academics",
                headers=headers,
                timeout=15
            )
            
            if response.status_code == 200:
                academic_data = response.json()
                return {
                    "success": True,
                    "data": academic_data,
                    "summary": self._generate_academic_summary(academic_data)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to fetch academic records: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Academic records fetch error: {str(e)}"
            }
    
    def fetch_attendance(self, token: str, organization_api: str, user_id: str, period: str = "current") -> Dict[str, Any]:
        """Fetch attendance data"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            params = {"period": period}
            response = self.session.get(
                f"{organization_api}/students/{user_id}/attendance",
                headers=headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 200:
                attendance_data = response.json()
                return {
                    "success": True,
                    "data": attendance_data,
                    "summary": self._generate_attendance_summary(attendance_data)
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to fetch attendance: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Attendance fetch error: {str(e)}"
            }
    
    def submit_request(self, token: str, organization_api: str, user_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a request to the organization"""
        try:
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "user_id": user_id,
                "request_type": request_data.get("type"),
                "description": request_data.get("description"),
                "priority": request_data.get("priority", "medium"),
                "category": request_data.get("category")
            }
            
            response = self.session.post(
                f"{organization_api}/requests",
                headers=headers,
                json=payload,
                timeout=15
            )
            
            if response.status_code in [200, 201]:
                result = response.json()
                return {
                    "success": True,
                    "request_id": result.get("id"),
                    "status": result.get("status"),
                    "message": "Request submitted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to submit request: {response.status_code}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Request submission error: {str(e)}"
            }
    
    def _generate_profile_summary(self, profile_data: Dict[str, Any]) -> str:
        """Generate human-readable profile summary"""
        summary_parts = []
        
        if profile_data.get("name"):
            summary_parts.append(f"Name: {profile_data['name']}")
        
        if profile_data.get("role"):
            summary_parts.append(f"Role: {profile_data['role']}")
        
        if profile_data.get("department"):
            summary_parts.append(f"Department: {profile_data['department']}")
        
        if profile_data.get("email"):
            summary_parts.append(f"Email: {profile_data['email']}")
        
        return " | ".join(summary_parts)
    
    def _generate_academic_summary(self, academic_data: Dict[str, Any]) -> str:
        """Generate academic records summary"""
        summary_parts = []
        
        if academic_data.get("gpa"):
            summary_parts.append(f"GPA: {academic_data['gpa']}")
        
        if academic_data.get("year"):
            summary_parts.append(f"Academic Year: {academic_data['year']}")
        
        if academic_data.get("major"):
            summary_parts.append(f"Major: {academic_data['major']}")
        
        if academic_data.get("credits_completed"):
            summary_parts.append(f"Credits: {academic_data['credits_completed']}")
        
        return " | ".join(summary_parts)
    
    def _generate_attendance_summary(self, attendance_data: Dict[str, Any]) -> str:
        """Generate attendance summary"""
        if attendance_data.get("percentage"):
            return f"Attendance: {attendance_data['percentage']}%"
        return "Attendance data available"