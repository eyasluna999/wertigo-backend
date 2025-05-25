import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from datetime import datetime
import logging
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('email_service')

# Email configuration
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USER = 'christianpacifico20@gmail.com'
EMAIL_PASSWORD = 'deoqsfkuocmczvzk'
EMAIL_FROM = 'WeTravel@gmail.com'

def send_ticket_email(recipient_email, ticket_id, travel_data):
    """
    Send an email to the user with their ticket details
    
    Args:
        recipient_email (str): User's email address
        ticket_id (str): The generated ticket ID
        travel_data (dict): The travel planning data
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        logger.info(f"Preparing to send email for ticket {ticket_id} to {recipient_email}")
        
        # Create message container
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f'Your Travel Ticket: {ticket_id}'
        msg['From'] = EMAIL_FROM
        msg['To'] = recipient_email
        
        # Create HTML email content
        html_content = generate_email_template(ticket_id, travel_data)
        
        # Attach HTML content
        part = MIMEText(html_content, 'html')
        msg.attach(part)
        
        logger.info("Connecting to SMTP server...")
        # Connect to SMTP server
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.ehlo()  # Can help with connection issues
        server.starttls(context=ssl.create_default_context())  # Secure the connection
        server.ehlo()  # Re-identify ourselves over TLS connection
        
        logger.info("Logging into email account...")
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        
        logger.info("Sending email...")
        # Send email
        server.sendmail(EMAIL_FROM, recipient_email, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"Authentication error with email provider: {e}")
        logger.error("Please check your email username and password/app password")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending email: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error sending email: {e}")
        return False

def generate_email_template(ticket_id, travel_data):
    """
    Generate modern HTML email content with travel details
    
    Args:
        ticket_id (str): The ticket ID
        travel_data (dict): The travel planning data
    
    Returns:
        str: HTML content for the email
    """
    # Get current date for footer
    current_date = datetime.now().strftime("%B %d, %Y")
    
    # Extract travel data
    destination = travel_data.get('destination', 'Not specified')
    travel_dates = travel_data.get('travel_dates', 'Not specified')
    travelers = travel_data.get('travelers', 1)
    budget = travel_data.get('budget', 'Not specified')
    budget_breakdown = travel_data.get('budget_breakdown', {})
    trip_summary = travel_data.get('trip_summary', {})
    
    # Get status information
    status = travel_data.get('status', 'active')
    status_message = travel_data.get('status_message', '')
    
    # Status styling
    status_colors = {
        'active': '#3b82f6',      # Modern Blue
        'confirmed': '#10b981',   # Modern Green
        'cancelled': '#ef4444',   # Modern Red
        'completed': '#8b5cf6',   # Modern Purple
        'pending': '#f59e0b'      # Modern Orange
    }
    
    status_color = status_colors.get(status, '#3b82f6')
    
    # Format budget to include PHP symbol if it's a number
    if isinstance(budget, (int, float)):
        budget = f"₱{budget:,.2f}"
    elif isinstance(budget, str) and budget.replace('₱', '').replace(',', '').isdigit():
        budget = f"₱{float(budget.replace('₱', '').replace(',', '')):,.2f}"
    
    # Generate budget breakdown HTML
    budget_breakdown_html = ""
    if budget_breakdown and any(budget_breakdown.values()):
        budget_breakdown_html = f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 25px; margin: 25px 0; color: white;">
            <h3 style="margin: 0 0 20px 0; font-size: 20px; font-weight: 600;">💰 Budget Breakdown</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
        """
        
        categories = {
            'accommodation': {'icon': '🏨', 'label': 'Accommodation'},
            'food': {'icon': '🍽️', 'label': 'Food & Dining'},
            'activities': {'icon': '🎯', 'label': 'Activities'},
            'transportation': {'icon': '🚗', 'label': 'Transportation'},
            'other': {'icon': '💳', 'label': 'Other Expenses'}
        }
        
        for category, amount in budget_breakdown.items():
            if category != 'destinations' and amount > 0:
                cat_info = categories.get(category, {'icon': '💰', 'label': category.title()})
                budget_breakdown_html += f"""
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 15px; text-align: center;">
                    <div style="font-size: 24px; margin-bottom: 8px;">{cat_info['icon']}</div>
                    <div style="font-size: 14px; margin-bottom: 5px; opacity: 0.9;">{cat_info['label']}</div>
                    <div style="font-size: 18px; font-weight: 600;">₱{amount:,}</div>
                </div>
                """
        
        budget_breakdown_html += """
            </div>
        </div>
        """
    
    # Generate itinerary HTML with modern design
    itinerary_html = ""
    if 'itinerary' in travel_data and travel_data['itinerary']:
        for day_index, day in enumerate(travel_data['itinerary']):
            day_num = day.get('day', day_index + 1)
            places = day.get('places', [])
            meals = day.get('meals', [])
            transportation = day.get('transportation', 'Not specified')
            estimated_cost = day.get('estimated_cost', 'Not specified')
            
            # Modern gradient colors for each day
            gradient_colors = [
                'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
                'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
            ]
            
            gradient = gradient_colors[day_index % len(gradient_colors)]
            
            itinerary_html += f"""
            <div style="margin-bottom: 30px; border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0,0,0,0.1);">
                <div style="background: {gradient}; color: white; padding: 20px;">
                    <h3 style="margin: 0; font-size: 24px; font-weight: 600;">📅 Day {day_num}</h3>
                    <p style="margin: 5px 0 0 0; opacity: 0.9;">Your adventure continues...</p>
                </div>
                
                <div style="background: white; padding: 25px;">
            """
            
            # Places section
            if places:
                itinerary_html += """
                <div style="margin-bottom: 25px;">
                    <h4 style="color: #1f2937; margin: 0 0 15px 0; font-size: 18px; display: flex; align-items: center;">
                        🏛️ <span style="margin-left: 8px;">Places to Visit</span>
                    </h4>
                    <div style="display: grid; gap: 15px;">
                """
                
                for place in places:
                    place_name = place.get('name', 'Unknown')
                    category = place.get('category', '')
                    description = place.get('description', '')
                    rating = place.get('rating', 0)
                    star_rating = '⭐' * int(rating) if rating else ''
                    
                    itinerary_html += f"""
                    <div style="background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%); border-radius: 12px; padding: 20px; border-left: 4px solid #3b82f6;">
                        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                            <h5 style="margin: 0; font-size: 18px; color: #1f2937; font-weight: 600;">{place_name}</h5>
                            <span style="background: #3b82f6; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;">{category}</span>
                        </div>
                        {f'<div style="margin: 8px 0; font-size: 16px;">{star_rating}</div>' if star_rating else ''}
                        {f'<p style="margin: 0; color: #6b7280; line-height: 1.5; font-size: 14px;">{description[:150]}{"..." if len(description) > 150 else ""}</p>' if description else ''}
                    </div>
                    """
                
                itinerary_html += """
                    </div>
                </div>
                """
            
            # Meals section
            if meals:
                itinerary_html += """
                <div style="margin-bottom: 25px;">
                    <h4 style="color: #1f2937; margin: 0 0 15px 0; font-size: 18px; display: flex; align-items: center;">
                        🍽️ <span style="margin-left: 8px;">Meal Suggestions</span>
                    </h4>
                    <div style="background: #fef3c7; border-radius: 12px; padding: 20px; border-left: 4px solid #f59e0b;">
                """
                
                for meal in meals:
                    meal_type = meal.get('type', '')
                    suggestion = meal.get('suggestion', '')
                    
                    itinerary_html += f"""
                        <div style="margin-bottom: 10px; display: flex; align-items: center;">
                            <span style="background: #f59e0b; color: white; padding: 4px 8px; border-radius: 6px; font-size: 12px; font-weight: 500; margin-right: 12px;">{meal_type}</span>
                            <span style="color: #92400e;">{suggestion}</span>
                        </div>
                    """
                
                itinerary_html += """
                    </div>
                </div>
                """
            
            # Transportation and cost
            itinerary_html += f"""
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;">
                    <div style="background: #ecfdf5; border-radius: 12px; padding: 15px; border-left: 4px solid #10b981;">
                        <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; font-weight: 600; display: flex; align-items: center;">
                            🚗 <span style="margin-left: 6px;">Transportation</span>
                        </h5>
                        <p style="margin: 0; color: #047857; font-size: 13px;">{transportation}</p>
                    </div>
                    <div style="background: #fef2f2; border-radius: 12px; padding: 15px; border-left: 4px solid #ef4444;">
                        <h5 style="margin: 0 0 8px 0; color: #1f2937; font-size: 14px; font-weight: 600; display: flex; align-items: center;">
                            💰 <span style="margin-left: 6px;">Estimated Cost</span>
                        </h5>
                        <p style="margin: 0; color: #dc2626; font-size: 13px; font-weight: 600;">{estimated_cost}</p>
                    </div>
                </div>
                </div>
            </div>
            """
    else:
        # Modern no-itinerary message
        itinerary_html = """
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); border-radius: 15px; padding: 40px; text-align: center; margin: 20px 0;">
            <div style="font-size: 48px; margin-bottom: 15px;">🗺️</div>
            <h3 style="color: #3730a3; margin: 0 0 10px 0;">Custom Itinerary</h3>
            <p style="color: #5b21b6; margin: 0;">Your personalized day-by-day itinerary will be added here. Contact us to customize your perfect trip!</p>
        </div>
        """
    
    # Destinations showcase
    destinations_html = ""
    if 'selected_destinations' in travel_data and travel_data['selected_destinations']:
        destinations = travel_data['selected_destinations'][:6]  # Show max 6 destinations
        
        destinations_html = f"""
        <div style="margin: 30px 0;">
            <h3 style="color: #1f2937; margin: 0 0 20px 0; font-size: 22px; font-weight: 600; text-align: center;">🎯 Your Selected Destinations</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
        """
        
        for dest in destinations:
            dest_name = dest.get('name', 'Unknown')
            dest_category = dest.get('category', '')
            dest_city = dest.get('city', '')
            dest_budget = dest.get('budget', 'N/A')
            
            # Format destination budget
            if isinstance(dest_budget, (int, float)):
                dest_budget = f"₱{dest_budget:,}"
            elif isinstance(dest_budget, str) and dest_budget.isdigit():
                dest_budget = f"₱{int(dest_budget):,}"
            
            destinations_html += f"""
            <div style="background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); border-top: 4px solid #3b82f6;">
                <h4 style="margin: 0 0 8px 0; color: #1f2937; font-size: 16px; font-weight: 600;">{dest_name}</h4>
                <div style="background: #eff6ff; color: #1d4ed8; padding: 4px 8px; border-radius: 6px; font-size: 12px; display: inline-block; margin-bottom: 10px;">{dest_category}</div>
                <p style="margin: 0 0 10px 0; color: #6b7280; font-size: 14px;">📍 {dest_city}</p>
                <p style="margin: 0; color: #059669; font-weight: 600; font-size: 14px;">💰 Budget: {dest_budget}</p>
            </div>
            """
        
        destinations_html += """
            </div>
        </div>
        """
    
    # Status message section
    status_message_html = ""
    if status_message:
        status_message_html = f"""
        <div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%); border-radius: 12px; padding: 20px; margin-bottom: 25px; border-left: 5px solid {status_color};">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; margin-right: 12px;">ℹ️</div>
                <p style="font-size: 16px; margin: 0; color: #1f2937; line-height: 1.5;">{status_message}</p>
            </div>
        </div>
        """
    
    # Trip summary stats
    trip_stats_html = ""
    if trip_summary:
        total_destinations = trip_summary.get('total_destinations', 0)
        duration = trip_summary.get('estimated_duration', 'N/A')
        categories = trip_summary.get('main_categories', [])
        
        trip_stats_html = f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); border-radius: 15px; padding: 25px; margin: 25px 0; color: white; text-align: center;">
            <h3 style="margin: 0 0 20px 0; font-size: 20px; font-weight: 600;">📊 Trip Overview</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 20px;">
                <div>
                    <div style="font-size: 32px; font-weight: 700; margin-bottom: 5px;">{total_destinations}</div>
                    <div style="font-size: 14px; opacity: 0.9;">Destinations</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 700; margin-bottom: 5px;">{duration}</div>
                    <div style="font-size: 14px; opacity: 0.9;">Duration</div>
                </div>
                <div>
                    <div style="font-size: 32px; font-weight: 700; margin-bottom: 5px;">{travelers}</div>
                    <div style="font-size: 14px; opacity: 0.9;">Travelers</div>
                </div>
            </div>
        </div>
        """

    # HTML template with modern design
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your WerTigo Travel Ticket</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        </style>
    </head>
    <body style="font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; color: #1f2937; max-width: 650px; margin: 0 auto; padding: 20px; background-color: #f9fafb;">
        
        <!-- Header with modern design -->
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 40px 30px; text-align: center; margin-bottom: 30px; color: white; position: relative; overflow: hidden;">
            <div style="position: absolute; top: -20px; right: -20px; width: 100px; height: 100px; background: rgba(255,255,255,0.1); border-radius: 50%; opacity: 0.6;"></div>
            <div style="position: absolute; bottom: -30px; left: -30px; width: 80px; height: 80px; background: rgba(255,255,255,0.1); border-radius: 50%; opacity: 0.4;"></div>
            
            <div style="position: relative; z-index: 1;">
                <h1 style="margin: 0 0 10px 0; font-size: 32px; font-weight: 700; letter-spacing: -0.5px;">✈️ WerTigo Travel</h1>
                <p style="margin: 0 0 20px 0; font-size: 16px; opacity: 0.9;">Your Personalized Travel Experience</p>
                
                <div style="background: rgba(255,255,255,0.2); border-radius: 12px; padding: 20px; margin-top: 25px; backdrop-filter: blur(10px);">
                    <p style="margin: 0 0 8px 0; font-size: 14px; opacity: 0.8; text-transform: uppercase; letter-spacing: 1px;">Ticket ID</p>
                    <p style="margin: 0; font-size: 28px; font-weight: 700; letter-spacing: 2px;">{ticket_id}</p>
                </div>
                
                <div style="margin-top: 20px;">
                    <span style="background: {status_color}; color: white; padding: 8px 16px; border-radius: 20px; font-weight: 600; text-transform: uppercase; font-size: 12px; letter-spacing: 0.5px;">
                        {status}
                    </span>
                </div>
            </div>
        </div>
        
        {status_message_html}
        
        <!-- Trip Details Card -->
        <div style="background: white; border-radius: 16px; padding: 30px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
            <h2 style="color: #1f2937; margin: 0 0 25px 0; font-size: 24px; font-weight: 600; display: flex; align-items: center;">
                🌍 <span style="margin-left: 10px;">Trip Details</span>
            </h2>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <div style="padding: 20px; background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); border-radius: 12px; border-left: 4px solid #f59e0b;">
                    <div style="font-size: 14px; color: #92400e; margin-bottom: 5px; font-weight: 500;">📍 Destination</div>
                    <div style="font-size: 18px; font-weight: 600; color: #78350f;">{destination}</div>
                </div>
                <div style="padding: 20px; background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 12px; border-left: 4px solid #3b82f6;">
                    <div style="font-size: 14px; color: #1e40af; margin-bottom: 5px; font-weight: 500;">📅 Travel Dates</div>
                    <div style="font-size: 18px; font-weight: 600; color: #1e3a8a;">{travel_dates}</div>
                </div>
                <div style="padding: 20px; background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%); border-radius: 12px; border-left: 4px solid #10b981;">
                    <div style="font-size: 14px; color: #047857; margin-bottom: 5px; font-weight: 500;">👥 Travelers</div>
                    <div style="font-size: 18px; font-weight: 600; color: #064e3b;">{travelers} person{'s' if travelers != 1 else ''}</div>
                </div>
                <div style="padding: 20px; background: linear-gradient(135deg, #fdf2f8 0%, #fce7f3 100%); border-radius: 12px; border-left: 4px solid #ec4899;">
                    <div style="font-size: 14px; color: #be185d; margin-bottom: 5px; font-weight: 500;">💰 Total Budget</div>
                    <div style="font-size: 18px; font-weight: 600; color: #9d174d;">{budget}</div>
                </div>
            </div>
        </div>
        
        {trip_stats_html}
        {budget_breakdown_html}
        {destinations_html}
        
        <!-- Itinerary Section -->
        <div style="background: white; border-radius: 16px; padding: 30px; margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e5e7eb;">
            <h2 style="color: #1f2937; margin: 0 0 25px 0; font-size: 24px; font-weight: 600; display: flex; align-items: center;">
                🗓️ <span style="margin-left: 10px;">Your Itinerary</span>
            </h2>
            {itinerary_html}
        </div>
        
        <!-- Footer -->
        <div style="text-align: center; margin-top: 40px; padding: 30px; background: linear-gradient(135deg, #1f2937 0%, #374151 100%); border-radius: 16px; color: white;">
            <div style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 10px 0; font-size: 20px; font-weight: 600;">Thank You for Choosing WerTigo! 🙏</h3>
                <p style="margin: 0; opacity: 0.8; font-size: 16px;">Your adventure awaits!</p>
            </div>
            
            <div style="border-top: 1px solid rgba(255,255,255,0.2); padding-top: 20px; font-size: 14px; opacity: 0.7;">
                <p style="margin: 0 0 5px 0;">Need help? Reply to this email or visit our support center</p>
                <p style="margin: 0;">Generated on {current_date}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html 