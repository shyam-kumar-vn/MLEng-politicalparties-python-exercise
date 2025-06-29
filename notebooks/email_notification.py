# Databricks notebook source
# MAGIC %md
# MAGIC # Email Notification for Model Promotion
# MAGIC 
# MAGIC This notebook sends email notifications when a model is promoted to production:
# MAGIC 1. Check if model promotion occurred
# MAGIC 2. Gather model promotion details
# MAGIC 3. Send email notification to stakeholders
# MAGIC 4. Log notification status

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup and Configuration

# COMMAND ----------

# DBTITLE 1,Import required libraries
import os
import json
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, List

from src.utils import get_widget_value, print_parameters

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("All components imported successfully")

# COMMAND ----------

# DBTITLE 1,Configure parameters
# Get parameters from workflow
CATALOG_NAME = get_widget_value("catalog_name", "mle_batch_catalog_2025_q2")
SCHEMA_NAME = get_widget_value("schema_name", "mle_shyamkumar_vn")
MODEL_NAME = get_widget_value("model_name", "political_party_classifier")
DBFS_BASE_PATH = get_widget_value("dbfs_base_path", "/dbfs/FileStore/shyamkumar.vn")
NOTIFICATION_EMAIL = get_widget_value("notification_email", "shyamkumar.vn@thoughtworks.com")
SMTP_SERVER = get_widget_value("smtp_server", "smtp.gmail.com")
SMTP_PORT = int(get_widget_value("smtp_port", "587"))

# Print parameters
params = {
    "Catalog": CATALOG_NAME,
    "Schema": SCHEMA_NAME,
    "Model": MODEL_NAME,
    "DBFS Base Path": DBFS_BASE_PATH,
    "Notification Email": NOTIFICATION_EMAIL,
    "SMTP Server": SMTP_SERVER,
    "SMTP Port": SMTP_PORT
}
print_parameters(params)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Model Promotion Status

# COMMAND ----------

# DBTITLE 1,Load promotion log and check status
# Load promotion log to check if model was promoted
promotion_log_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_promotion_log.json"

try:
    with open(promotion_log_path, 'r') as f:
        promotion_log = json.load(f)
    
    promotion_occurred = promotion_log.get('promotion_occurred', False)
    promotion_timestamp = promotion_log.get('timestamp', '')
    promotion_details = promotion_log.get('promotion_details', {})
    
    print(f"Promotion occurred: {promotion_occurred}")
    print(f"Promotion timestamp: {promotion_timestamp}")
    
except FileNotFoundError:
    print(f"Promotion log not found at {promotion_log_path}")
    promotion_occurred = False
    promotion_details = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Email Configuration

# COMMAND ----------

# DBTITLE 1,Configure email settings
# Email configuration
EMAIL_CONFIG = {
    'smtp_server': SMTP_SERVER,
    'smtp_port': SMTP_PORT,
    'sender_email': 'ml-notifications@thoughtworks.com',
    'recipients': [NOTIFICATION_EMAIL],
    'subject_prefix': '[ML Model Promotion]'
}

# Try to get SMTP credentials from Databricks secrets
try:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils()
    
    # Get SMTP credentials from secrets
    smtp_username = dbutils.secrets.get(scope="email-notifications", key="smtp-username")
    smtp_password = dbutils.secrets.get(scope="email-notifications", key="smtp-password")
    
    EMAIL_CONFIG['smtp_username'] = smtp_username
    EMAIL_CONFIG['smtp_password'] = smtp_password
    
    print("SMTP credentials loaded from Databricks secrets")
    
except Exception as e:
    print(f"Could not load SMTP credentials from secrets: {e}")
    print("Will try environment variables or use default configuration")
    
    # Fallback to environment variables
    EMAIL_CONFIG['smtp_username'] = os.getenv('SMTP_USERNAME', '')
    EMAIL_CONFIG['smtp_password'] = os.getenv('SMTP_PASSWORD', '')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Email Content

# COMMAND ----------

# DBTITLE 1,Create email content based on promotion status
def create_email_content(promotion_occurred: bool, promotion_details: Dict[str, Any]) -> str:
    """Create email content based on promotion status"""
    
    if promotion_occurred:
        # Model was promoted
        subject = f"{EMAIL_CONFIG['subject_prefix']} Model Promoted to Production"
        
        # Extract promotion details
        new_model_version = promotion_details.get('new_model_version', 'Unknown')
        old_model_version = promotion_details.get('old_model_version', 'None')
        improvement = promotion_details.get('improvement', 0.0)
        new_f1_score = promotion_details.get('new_f1_score', 0.0)
        old_f1_score = promotion_details.get('old_f1_score', 0.0)
        
        html_content = f"""
        <html>
        <body>
        <h2>🚀 Model Promotion Notification</h2>
        <p><strong>Model:</strong> {CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}</p>
        <p><strong>Promotion Time:</strong> {promotion_timestamp}</p>
        
        <h3>Promotion Details</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #f2f2f2;">
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>New Model Version</td>
                <td>{new_model_version}</td>
            </tr>
            <tr>
                <td>Previous Model Version</td>
                <td>{old_model_version}</td>
            </tr>
            <tr>
                <td>F1 Score Improvement</td>
                <td style="color: green;">+{improvement:.4f}</td>
            </tr>
            <tr>
                <td>New Model F1 Score</td>
                <td>{new_f1_score:.4f}</td>
            </tr>
            <tr>
                <td>Previous Model F1 Score</td>
                <td>{old_f1_score:.4f}</td>
            </tr>
        </table>
        
        <h3>Next Steps</h3>
        <ul>
            <li>Monitor model performance in production</li>
            <li>Check serving endpoint health</li>
            <li>Review batch inference results</li>
            <li>Update model documentation if needed</li>
        </ul>
        
        <h3>Contact Information</h3>
        <p>For questions or issues regarding this model promotion, please contact:</p>
        <ul>
            <li>ML Team: ml-team@thoughtworks.com</li>
            <li>DevOps Team: devops@thoughtworks.com</li>
        </ul>
        
        <hr>
        <p><em>This is an automated notification from the ML pipeline.</em></p>
        </body>
        </html>
        """
        
    else:
        # Model was not promoted
        subject = f"{EMAIL_CONFIG['subject_prefix']} Model Promotion Skipped"
        
        # Extract details
        new_f1_score = promotion_details.get('new_f1_score', 0.0)
        old_f1_score = promotion_details.get('old_f1_score', 0.0)
        threshold = promotion_details.get('threshold', 0.01)
        
        html_content = f"""
        <html>
        <body>
        <h2>⚠️ Model Promotion Skipped</h2>
        <p><strong>Model:</strong> {CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}</p>
        <p><strong>Evaluation Time:</strong> {promotion_timestamp}</p>
        
        <h3>Promotion Decision</h3>
        <p>The new model was <strong>not promoted</strong> to production because it did not meet the promotion criteria.</p>
        
        <h3>Performance Comparison</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr style="background-color: #f2f2f2;">
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>New Model F1 Score</td>
                <td>{new_f1_score:.4f}</td>
            </tr>
            <tr>
                <td>Current Production F1 Score</td>
                <td>{old_f1_score:.4f}</td>
            </tr>
            <tr>
                <td>Improvement</td>
                <td style="color: red;">{new_f1_score - old_f1_score:+.4f}</td>
            </tr>
            <tr>
                <td>Promotion Threshold</td>
                <td>{threshold:.4f}</td>
            </tr>
        </table>
        
        <h3>Next Steps</h3>
        <ul>
            <li>Review model training process</li>
            <li>Consider feature engineering improvements</li>
            <li>Investigate potential data quality issues</li>
            <li>Retrain with different hyperparameters if needed</li>
        </ul>
        
        <h3>Contact Information</h3>
        <p>For questions regarding this model evaluation, please contact:</p>
        <ul>
            <li>ML Team: ml-team@thoughtworks.com</li>
            <li>Data Science Team: datascience@thoughtworks.com</li>
        </ul>
        
        <hr>
        <p><em>This is an automated notification from the ML pipeline.</em></p>
        </body>
        </html>
        """
    
    return subject, html_content

# Generate email content
subject, html_content = create_email_content(promotion_occurred, promotion_details)

print(f"Email subject: {subject}")
print(f"Email content generated successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Send Email Notification

# COMMAND ----------

# DBTITLE 1,Send email notification
def send_email_notification(subject: str, html_content: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Send email notification using SMTP"""
    
    try:
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config['sender_email']
        msg['To'] = ', '.join(config['recipients'])
        
        # Attach HTML content
        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)
        
        # Connect to SMTP server
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()  # Enable TLS
        
        # Login if credentials provided
        if config.get('smtp_username') and config.get('smtp_password'):
            server.login(config['smtp_username'], config['smtp_password'])
            logger.info("SMTP authentication successful")
        
        # Send email
        text = msg.as_string()
        server.sendmail(config['sender_email'], config['recipients'], text)
        server.quit()
        
        logger.info(f"Email sent successfully to {config['recipients']}")
        
        return {
            'success': True,
            'recipients': config['recipients'],
            'subject': subject,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Send email notification
email_result = send_email_notification(subject, html_content, EMAIL_CONFIG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Notification Status

# COMMAND ----------

# DBTITLE 1,Save notification log
# Create notification log
notification_log = {
    'notification_timestamp': datetime.now().isoformat(),
    'promotion_occurred': promotion_occurred,
    'email_sent': email_result['success'],
    'email_recipients': EMAIL_CONFIG['recipients'],
    'email_subject': subject,
    'promotion_details': promotion_details
}

if not email_result['success']:
    notification_log['email_error'] = email_result.get('error', 'Unknown error')

# Save notification log
notification_log_path = f"{DBFS_BASE_PATH}/{SCHEMA_NAME}_email_notification_log.json"
with open(notification_log_path, 'w') as f:
    json.dump(notification_log, f, indent=2)

print(f"Notification log saved to: {notification_log_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Display Notification Summary

# COMMAND ----------

# DBTITLE 1,Show notification summary
print("="*80)
print("EMAIL NOTIFICATION SUMMARY")
print("="*80)
print(f"Model: {CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}")
print(f"Promotion Occurred: {promotion_occurred}")
print(f"Email Sent: {email_result['success']}")
print(f"Recipients: {', '.join(EMAIL_CONFIG['recipients'])}")
print(f"Subject: {subject}")

if promotion_occurred:
    print(f"\nPromotion Details:")
    print(f"  New Model Version: {promotion_details.get('new_model_version', 'Unknown')}")
    print(f"  Previous Model Version: {promotion_details.get('old_model_version', 'None')}")
    print(f"  F1 Score Improvement: +{promotion_details.get('improvement', 0.0):.4f}")
else:
    print(f"\nPromotion Skipped:")
    print(f"  New Model F1 Score: {promotion_details.get('new_f1_score', 0.0):.4f}")
    print(f"  Current Production F1 Score: {promotion_details.get('old_f1_score', 0.0):.4f}")
    print(f"  Improvement: {promotion_details.get('new_f1_score', 0.0) - promotion_details.get('old_f1_score', 0.0):+.4f}")

if not email_result['success']:
    print(f"\nEmail Error: {email_result.get('error', 'Unknown error')}")

print(f"\nLog Files:")
print(f"  Promotion Log: {promotion_log_path}")
print(f"  Notification Log: {notification_log_path}")
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Email Notification Complete!
# MAGIC 
# MAGIC The email notification process has successfully:
# MAGIC - Checked model promotion status
# MAGIC - Generated appropriate email content
# MAGIC - Sent notification to stakeholders
# MAGIC - Logged notification details
# MAGIC 
# MAGIC Stakeholders have been notified of the model promotion decision! 