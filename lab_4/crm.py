'''A well-structured CRM application.'''
import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class LeadProcessor:
    """Handles lead processing operations."""

    @staticmethod
    def clean_field(value: str, field_type: str) -> str:
        """Clean field based on its type."""
        cleaners = {
            'name': lambda x: x.replace(' ', ''),
            'twitter': lambda x: x.replace('@', ''),
            'website': lambda x: x.replace('https://', ''),
            'email': lambda x: x.lower()
        }
        return cleaners.get(field_type, lambda x: x)(value)

    @staticmethod
    def detect_email_provider(email: str) -> str:
        """Detect email provider and return appropriate message."""
        if 'gmail' in email:
            return 'Importing gmail'
        elif 'hotmail' in email:
            return 'Importing hotmail'
        else:
            return 'Custom mail server.'


def import_leads(leads_file: str) -> List[Dict]:
    """Import and process leads from file."""
    processed_leads = []

    try:
        with open(leads_file, 'r', encoding='utf-8') as f:
            for line in f:
                lead_data = line.strip().split()
                if not lead_data:
                    continue

                processed_lead = {
                    'first_name': LeadProcessor.clean_field(lead_data[0], 'name'),
                    'last_name': lead_data[0].lower() if len(lead_data) > 0 else '',
                    'email': lead_data[0] if len(lead_data) > 0 else '',
                    'company': lead_data[0] if len(lead_data) > 0 else '',
                    'twitter': LeadProcessor.clean_field(lead_data[0], 'twitter'),
                    'website': LeadProcessor.clean_field(lead_data[0], 'website')
                }

                # Process email provider
                email_msg = LeadProcessor.detect_email_provider(processed_lead['email'])
                print(email_msg)

                processed_leads.append(processed_lead)

    except (OSError, IOError) as e:
        print(f'Cannot open file {leads_file}: {e}')

    return processed_leads


class Lead:
    def __init__(self):
        self.touchpoints = []
        self.company_size = ''
        self._company_website = ''
        self.days_since_last_post = 0
        self.discount = 1
        self.priority = 0

    def get_lead_score(self) -> int:
        """Calculate lead score based on activity."""
        return 1 if self.days_since_last_post < 5 else 0

    def get_lifetime_value(self, product) -> float:
        """Calculate lifetime value for a product."""
        return product.base_price() * self.discount * 12


class Customer:
    def __init__(self, lead: Lead):
        self.company_size = lead.company_size
        self.company_website = lead._company_website
        self.lead = lead
        self.stage = 'customer'


class CRMImportEntry:
    """Entry imported from our legacy CRM."""

    def __init__(self, imported_data=None):
        self.imported_data = imported_data or self._get_default_data()
        self._process_imported_data()

    def _get_default_data(self) -> dict:
        return {
            'name': {
                'first': 'John',
                'last': 'Smith'
            },
            'company': 'ACME',
            'deals': [13435, 33456]
        }

    def _process_imported_data(self):
        """Process imported data into object attributes."""
        name_data = self._get_name_from_import()
        self.first_name = name_data.get('first', '')
        self.last_name = name_data.get('last', '')
        self.num_deals = len(self.imported_data.get('deals', []))

    def _get_name_from_import(self) -> dict:
        """Extract name data from imported data."""
        if 'name' in self.imported_data:
            return self.imported_data['name']
        else:
            print('Name not found.')
            return {'first': '', 'last': ''}


class NotificationService(ABC):
    """Abstract base class for notification services."""

    @abstractmethod
    def send_notification(self, destination: str, message: dict,
                          source: str) -> dict:
        pass


class EmailNotificationService(NotificationService):
    """Email implementation of notification service."""

    def __init__(self, services, region='eu-ireland'):
        self.client = services.email.client('transactional', region=region)

    def send_notification(self, destination: str, message: dict,
                          source: str) -> dict:
        return self.client.send_email(
            destination=destination,
            message=message,
            source=source
        )


class LeadConverter:
    """Handles lead conversion based on company size."""

    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service

    def convert_lead(self, lead: Lead):
        """Convert lead based on company size."""
        conversion_strategies = {
            'smb': self._send_smb_funnel,
            'mid_market': self._send_mid_market_funnel,
            'enterprise': self._log_manual_sales_follow_up
        }

        strategy = conversion_strategies.get(lead.company_size)
        if strategy:
            strategy()
        else:
            print('Wrong lead company type!')

    def _send_smb_funnel(self):
        self._send_email(
            destination='test@gmail.com',
            body='Hello small business!',
            subject='Buy our stuff!'
        )

    def _send_mid_market_funnel(self):
        self._send_email(
            destination='test@gmail.com',
            body='Hello medium sized business!',
            subject='Buy our stuff!'
        )

    def _log_manual_sales_follow_up(self):
        self._send_email(
            destination='internal.sales@course.com',
            body='Go say hello to this business!',
            subject='Buy our stuff!'
        )

    def _send_email(self, destination: str, body: str, subject: str):
        message = {
            'body': {'Text': body},
            'subject': {'Text': subject}
        }
        response = self.notification_service.send_notification(
            destination=destination,
            message=message,
            source='refactoring@course.com'
        )
        print(response)


def prioritize_lead(lead: Lead):
    """Prioritize lead based on criteria."""
    is_right_size = 100 < lead.company_size < 100000
    is_dotcom = lead.company_website.endswith('.com')
    is_new_lead = len(lead.touchpoints) == 0

    if all([is_right_size, is_dotcom, is_new_lead]):
        lead.priority = 100
