'''A well-structured CRM application.'''
import os
import sys
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
    touchpoints = []
    company_size = ''
    _company_website = ''
    days_since_last_post = 0
    discount = 1

    def get_lead_score(self):
        return 1 if self.is_active() else 0

    # 3.2: a method that should be merged into get_lead_score
    def is_active(self):
        return self.days_since_last_post < 5

    # 3.2: mrr should be inlined to the return statement
    def get_lifetime_value(self, product):
        mrr = product.base_price() * self.discount
        return mrr * 12


class Customer:
    company_size = ''
    lead = Lead()
    company_website = ''

    def __init__(self, lead):
        # One class uses the internal fields and methods of another class.
        self.company_size = lead.company_size
        self.company_website = lead._company_website
        self.lead = lead


# 3.3 Replace complex expressions with inner function calls
class CRMImportEntry:
    '''Entry imported from our legacy CRM.'''
    def __init__(self):
        imported_data = {
            'name': {
                'first': 'John',
                'last': 'Smith'
            },
            'company': 'ACME',
            'deals': [13435, 33456]
        }

        def get_name_from_import(data):
            if 'name' in data:
                return data['name']
            else:
                print('Name not found.')
                return dict(first='', last='')

        self.first_name = get_name_from_import(imported_data).get('first', '')
        self.last_name = get_name_from_import(imported_data).get('last', '')
        self.num_deals = len(imported_data.get('deals', []))


def convert_lead(lead):
    if lead.company_size == 'smb':
        send_smb_funnel()
    elif lead.company_size == 'mid_market':
        send_mid_market_funnel()
    elif lead.company_size == 'enterprise':
        log_manual_sales_follow_up()
    else:
        print('Wrong lead company type!')


def send_smb_funnel(services=''):
    client = services.email.client('transactional', region='eu-ireland')
    response = client.send_email(
        destination='test@gmail.com',
        message={
            'body': {'Text': {'Hello small business!'}},
            'subject': {'Text': {'Buy our stuff!'}}
        },
        source='refactoring@course.com'
    )
    print(response)


def send_mid_market_funnel(services=''):
    client = services.email.client('transactional', region='eu-ireland')
    response = client.send_email(
        destination='test@gmail.com',
        message={
            'body': {'Text': {'Hello medium sized business!'}},
            'subject': {'Text': {'Buy our stuff!'}}
        },
        source='refactoring@course.com'
    )
    print(response)


def log_manual_sales_follow_up(services=''):
    client = services.email.client('transactional', region='eu-ireland')
    response = client.send_email(
        destination='internal.sales@course.com',
        message={
            'body': {'Text': {'Go say hello to this business!'}},
            'subject': {'Text': {'Buy our stuff!'}}
        },
        source='refactoring@course.com'
    )
    print(response)


# 3.4
def prioritize_lead(lead):
    is_right_size = (lead.company_size > 100) and (lead.company_size < 100000)
    is_dotcom = lead.company_website.endswith('.com')
    is_new_lead = len(lead.touchpoints) == 0
    if is_right_size and is_dotcom and is_new_lead:
        lead.priority = 100
