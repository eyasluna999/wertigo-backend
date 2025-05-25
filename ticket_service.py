from flask import Blueprint, jsonify, request
import database as db
import logging
from datetime import datetime
import random
import string
from email_service import send_ticket_email

# Configure logging
logger = logging.getLogger(__name__)

# Create blueprint
bp = Blueprint('ticket_service', __name__, url_prefix='/api')

@bp.route('/create_ticket', methods=['POST'])
def create_ticket():
    """
    Create a new ticket for tracking an itinerary.
    
    Expected JSON input:
    {
        "email": "user@example.com",
        "trip_id": "12345" or null,
        "itinerary": {...} (Optional, full itinerary data)
    }
    
    Returns:
    {
        "ticket_id": "WTO-ABC123",
        "status": "created"
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        email = data.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Generate a unique ticket ID in the format "WTO-ABC123"
        # Using a combination of letters and numbers for security
        random_chars = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        ticket_id = f"WTO-{random_chars}"
        
        # Get trip ID or itinerary
        trip_id = data.get('trip_id')
        itinerary = data.get('itinerary')
        
        if not trip_id and not itinerary:
            return jsonify({"error": "Either trip_id or itinerary must be provided"}), 400
        
        # If trip_id is provided, get the trip details
        travel_data = itinerary
        if trip_id and not itinerary:
            travel_data = db.get_trip_by_id(trip_id)
            if not travel_data:
                return jsonify({"error": f"Trip with ID {trip_id} not found"}), 404
            
        # Create ticket in database
        ticket = {
            "ticket_id": ticket_id,
            "email": email,
            "trip_id": trip_id,
            "itinerary": travel_data,
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Save ticket to database
        result = db.create_ticket(ticket)
        
        if "error" in result:
            return jsonify(result), 500
        
        # Send email notification with ticket details
        try:
            email_sent = send_ticket_email(email, ticket_id, travel_data)
            email_status = "Email notification sent." if email_sent else "Email notification failed to send."
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            email_status = "Email notification failed to send."
        
        # Return the ticket ID to the user
        return jsonify({
            "ticket_id": ticket_id,
            "status": "created",
            "email_status": email_status,
            "message": "Your ticket has been created. You can use this ticket ID to track your itinerary."
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating ticket: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/tickets/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    """
    Get ticket details by ticket ID.
    
    URL Parameters:
    - ticket_id: The ticket ID to look up
    
    Query Parameters (optional):
    - email: Email address for validation
    
    Returns the ticket details if found.
    """
    try:
        # Get email from query parameters (optional for validation)
        email = request.args.get('email')
        
        # Look up ticket in database
        ticket = db.get_ticket(ticket_id)
        
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404
            
        # If email is provided, validate that it matches
        if email and ticket['email'] != email:
            return jsonify({"error": "Email does not match ticket record"}), 403
            
        # Get trip details if trip_id is available
        trip_data = None
        if ticket.get('trip_id'):
            trip_data = db.get_trip_by_id(ticket['trip_id'])
        
        # Use ticket's itinerary if available
        itinerary_data = ticket.get('itinerary')
            
        # Create a more comprehensive response
        response = {
            "ticket": {
                "ticket_id": ticket['ticket_id'],
                "status": ticket['status'],
                "created_at": ticket['created_at'],
                "updated_at": ticket['updated_at'],
                # Include masked email for privacy
                "email_masked": mask_email(ticket['email']),
                # Include trip details if available
                "trip": trip_data or itinerary_data or {}
            }
        }
        
        # Add more ticket metadata
        if trip_data or itinerary_data:
            travel_info = trip_data or itinerary_data
            response["ticket"]["destination"] = travel_info.get('destination', 'Not specified')
            response["ticket"]["travel_dates"] = travel_info.get('travel_dates', 'Not specified')
            response["ticket"]["travelers"] = travel_info.get('travelers', 1)
            
            # Calculate days until trip - Fixed to handle 'To be determined'
            try:
                travel_dates = travel_info.get('travel_dates')
                if travel_dates and travel_dates not in ['To be determined', 'Not specified', None, '']:
                    dates = travel_dates.split(' to ')
                    if len(dates) > 0 and dates[0].strip():
                        # Try to parse different date formats
                        date_str = dates[0].strip()
                        start_date = None
                        
                        # Try different date formats
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
                        for fmt in date_formats:
                            try:
                                start_date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if start_date:
                            today = datetime.now()
                            days_until = (start_date - today).days
                            if days_until >= 0:
                                response["ticket"]["days_until_trip"] = days_until
                        else:
                            logger.warning(f"Could not parse travel date: {date_str}")
            except Exception as e:
                logger.error(f"Error calculating days until trip: {e}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error retrieving ticket: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/tickets', methods=['GET'])
def get_tickets_by_email():
    """
    Get all tickets associated with an email address.
    
    Query Parameters:
    - email: Email address to look up tickets for
    
    Returns a list of tickets.
    """
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email parameter is required"}), 400
            
        # Get tickets from database
        tickets = db.get_tickets_by_email(email)
        
        # Enhance ticket data with additional information
        enhanced_tickets = []
        for ticket in tickets:
            # Get basic ticket info
            ticket_info = {
                "ticket_id": ticket['ticket_id'],
                "status": ticket['status'],
                "created_at": ticket['created_at'],
                "updated_at": ticket['updated_at']
            }
            
            # Add destination and date information if available
            travel_info = ticket.get('itinerary') or {}
            if ticket.get('trip_id'):
                try:
                    trip_data = db.get_trip_by_id(ticket['trip_id'])
                    if trip_data:
                        travel_info = trip_data
                except Exception as e:
                    logger.error(f"Error fetching trip data: {e}")
            
            # Add summary information
            ticket_info["destination"] = travel_info.get('destination', 'Not specified')
            ticket_info["travel_dates"] = travel_info.get('travel_dates', 'Not specified')
            
            # Calculate days until trip - Fixed to handle 'To be determined'
            try:
                travel_dates = travel_info.get('travel_dates')
                if travel_dates and travel_dates not in ['To be determined', 'Not specified', None, '']:
                    dates = travel_dates.split(' to ')
                    if len(dates) > 0 and dates[0].strip():
                        date_str = dates[0].strip()
                        start_date = None
                        
                        # Try different date formats
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
                        for fmt in date_formats:
                            try:
                                start_date = datetime.strptime(date_str, fmt)
                                break
                            except ValueError:
                                continue
                        
                        if start_date:
                            today = datetime.now()
                            days_until = (start_date - today).days
                            if days_until >= 0:
                                ticket_info["days_until_trip"] = days_until
            except Exception as e:
                logger.error(f"Error calculating days until trip: {e}")
            
            enhanced_tickets.append(ticket_info)
        
        # Return list of tickets
        return jsonify({
            "tickets": enhanced_tickets
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving tickets: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/tickets/<ticket_id>/delete', methods=['DELETE'])
def delete_ticket(ticket_id):
    """
    Delete a ticket by ticket ID.
    
    URL Parameters:
    - ticket_id: The ticket ID to delete
    
    Query Parameters:
    - email: Email address for verification
    
    Returns success or error message.
    """
    try:
        # Get email from query parameters for verification
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email parameter is required for verification"}), 400
            
        # Delete ticket from database with email verification
        result = db.delete_ticket(ticket_id, email)
        
        if "error" in result:
            return jsonify(result), 404 if "not found" in result["error"].lower() else 400
            
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error deleting ticket: {e}")
        return jsonify({"error": str(e)}), 500

@bp.route('/tickets/<ticket_id>/update', methods=['PUT'])
def update_ticket_status(ticket_id):
    """
    Update a ticket's status and send an email notification.
    
    URL Parameters:
    - ticket_id: The ticket ID to update
    
    Expected JSON input:
    {
        "status": "confirmed" | "cancelled" | "completed",
        "email": "user@example.com" (optional, for verification)
    }
    
    Returns success or error message.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get the new status
        new_status = data.get('status')
        if not new_status:
            return jsonify({"error": "Status is required"}), 400
            
        # Validate status value
        valid_statuses = ['active', 'confirmed', 'cancelled', 'completed', 'pending']
        if new_status not in valid_statuses:
            return jsonify({"error": f"Invalid status. Must be one of: {', '.join(valid_statuses)}"}), 400
        
        # Get email for verification (optional)
        email = data.get('email')
        
        # Get the current ticket
        ticket = db.get_ticket(ticket_id)
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404
            
        # Verify email if provided
        if email and ticket['email'] != email:
            return jsonify({"error": "Email does not match ticket record"}), 403
        
        # Update the ticket status
        result = db.update_ticket_status(ticket_id, new_status)
        
        if "error" in result:
            return jsonify(result), 400
            
        # Get trip data for the email notification
        travel_data = ticket.get('itinerary') or {}
        if ticket.get('trip_id'):
            try:
                trip_data = db.get_trip_by_id(ticket['trip_id'])
                if trip_data:
                    travel_data = trip_data
            except Exception as e:
                logger.error(f"Error fetching trip data: {e}")
        
        # Send email notification about status change
        try:
            email_sent = send_ticket_email(ticket['email'], ticket_id, travel_data, status_update=new_status)
            email_status = "Email notification sent." if email_sent else "Email notification failed to send."
        except Exception as e:
            logger.error(f"Error sending status update email: {e}")
            email_status = "Email notification failed to send."
        
        return jsonify({
            "message": f"Ticket status updated to '{new_status}'",
            "email_status": email_status
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating ticket status: {e}")
        return jsonify({"error": str(e)}), 500

def mask_email(email):
    """
    Mask an email address for privacy.
    
    Example: user@example.com becomes u***@e******.com
    """
    if not email or '@' not in email:
        return "***@***.***"
    
    parts = email.split('@')
    username = parts[0]
    domain = parts[1]
    
    # Mask username (show first character + asterisks)
    masked_username = username[0] + '*' * (len(username) - 1) if len(username) > 1 else '*'
    
    # Mask domain (show first character + asterisks + last part)
    domain_parts = domain.split('.')
    if len(domain_parts) >= 2:
        masked_domain = domain_parts[0][0] + '*' * (len(domain_parts[0]) - 1) + '.' + domain_parts[-1]
    else:
        masked_domain = domain[0] + '*' * (len(domain) - 1)
    
    return f"{masked_username}@{masked_domain}" 