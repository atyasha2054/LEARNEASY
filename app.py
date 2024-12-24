from flask import Flask, redirect, render_template, request, make_response, session, abort, jsonify, url_for
import secrets
from functools import wraps
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import timedelta
import os
import joblib
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
model = joblib.load('D:/flask website/Flask-Firebase-Template/static_xgboost_college_model.pkl')


app.secret_key = os.getenv('SECRET_KEY')

# Configure session cookie settings
app.config['SESSION_COOKIE_SECURE'] = True  # Ensure cookies are sent over HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)  # Adjust session expiration as needed
app.config['SESSION_REFRESH_EACH_REQUEST'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # Can be 'Strict', 'Lax', or 'None'


# Firebase Admin SDK setup
cred = credentials.Certificate("firebase-auth.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# books = [
#     {
#         "title": "Introduction to Algorithms",
#         "author": "Cormen, Leiserson, Rivest, Stein",
#         "isbn": "9780262033848",
#         "released": "Jul 31, 2009",
#         "publisher": "MIT Press",
#         "format": "Hardcover",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/81TgbrThO8L.jpg"
#     },
#     {
#         "title": "Clean Code: A Handbook of Agile Software Craftsmanship",
#         "author": "Robert C. Martin",
#         "isbn": "9780132350884",
#         "released": "Aug 01, 2008",
#         "publisher": "Prentice Hall",
#         "format": "Paperback",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/41jEbK-jG+L._SX258_BO1,204,203,200_.jpg"
#     },
#     {
#         "title": "Artificial Intelligence: A Modern Approach",
#         "author": "Stuart Russell, Peter Norvig",
#         "isbn": "9780136042594",
#         "released": "Dec 11, 2009",
#         "publisher": "Pearson",
#         "format": "Hardcover",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/71G6T1yIZaL.jpg"
#     },
#     {
#         "title": "Operating System Concepts",
#         "author": "Abraham Silberschatz, Peter B. Galvin, Greg Gagne",
#         "isbn": "9781118063330",
#         "released": "Feb 15, 2011",
#         "publisher": "Wiley",
#         "format": "Paperback",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/81Jb5r6-WHL.jpg"
#     },
#     {
#         "title": "The Design of Everyday Things",
#         "author": "Don Norman",
#         "isbn": "9780465050659",
#         "released": "Nov 05, 2013",
#         "publisher": "Basic Books",
#         "format": "Paperback",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/71HMyqG6MRL.jpg"
#     },
#     {
#         "title": "Structure and Interpretation of Computer Programs",
#         "author": "Harold Abelson, Gerald Jay Sussman",
#         "isbn": "9780262510875",
#         "released": "Sep 01, 1996",
#         "publisher": "MIT Press",
#         "format": "Hardcover",
#         "image": "https://images-na.ssl-images-amazon.com/images/I/71bLBd8VSDL.jpg"
#     }
# ]
# books = [
#     {
#         "id": 1,
#         "title": "Engineering Mathematics",
#         "author": "K.A. Stroud", 
#         "isbn": "978-1137031204",
#         "released": "2013",
#         "publisher": "Palgrave Macmillan",
#         "format": "Hardcover",
#         "image": "https://example.com/engineering-math.jpg",
#         "main_image_url": "https://m.media-amazon.com/images/I/61Iz2yy2CKL.jpg",
#         "qr_code_url": "https://api.qrserver.com/v1/create-qr-code/?data=https://example.com&size=100x100",
#         "card_title": "Mastering Mathematics",
#         "card_subtitle": "Third Edition",
#     },
#     {
#         "id": 2,
#         "title": "Introduction to Algorithms",
#         "author": "Thomas H. Cormen",
#         "isbn": "978-0262033848",
#         "released": "2009",
#         "publisher": "MIT Press",
#         "format": "Hardcover",
#         "image": "https://example.com/algorithms.jpg",
#         "main_image_url": "https://m.media-amazon.com/images/I/61Iz2yy2CKL.jpg",
#         "qr_code_url": "https://api.qrserver.com/v1/create-qr-code/?data=https://example.com&size=100x100",
#         "card_title": "Introduction to Algorithms",
#         "card_subtitle": "Fourth Edition",
#     },
# ]
books = [
    {
        "id": 1,
        "title": "An Introduction to Mechanical Engineering",
        "author": "Jonathan Wickert",
        "isbn": "978-1305635135",
        "released": "2016",
        "publisher": "Cengage Learning",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/x7zSCWQ/11.jpg",
        "qr_code_url": "https://i.ibb.co/t85FWqQ/8.png",
        "card_title": "Fundamentals of Engineering",
        "card_subtitle": "Fourth Edition"
    },
    {
        "id": 2,
        "title": "Cinderella",
        "author": "Charles Perrault",
        "isbn": "978-1402729801",
        "released": "2007",
        "publisher": "Sterling Publishing",
        "format": "Hardcover",
        "main_image_url": "https://i.ibb.co/FJtcFwW/13.jpg",
        "qr_code_url": "https://i.ibb.co/MCbMrCg/2.png",
        "card_title": "Classic Fairy Tale",
        "card_subtitle": "Illustrated Edition"
    },
    {
        "id": 3,
        "title": "Engineering Mathematics with John Bird FreeBook",
        "author": "John Bird",
        "isbn": "978-1138672673",
        "released": "2017",
        "publisher": "Routledge",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/swm4rXD/10.jpg",
        "qr_code_url": "https://i.ibb.co/gjtDrxr/math.png",
        "card_title": "Mastering Engineering Math",
        "card_subtitle": "Essential Edition"
    },
    {
        "id": 4,
        "title": "Engineering Physics I & II",
        "subtitle": "Diploma Course in Engineering - First and Second Semester",
        "author": "Government of Tamilnadu",
        "isbn": "",
        "released": "2015",
        "publisher": "Directorate of Technical Education, Government of Tamilnadu",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/NyMH4Gj/Screenshot-2024-12-24-005708.png",
        "qr_code_url": "https://i.ibb.co/PtTHqkj/1.png",
        "card_title": "Engineering Physics",
        "card_subtitle": "First and Second Semester"
    },
    {
        "id": 5,
        "title": "Engineering Chemistry",
        "author": "Jain & Jain",
        "isbn": "9352160002",
        "released": "January 1, 2015",
        "publisher": "DHANPAT RAI",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/bbxvjyD/9.jpg",
        "qr_code_url": "https://i.ibb.co/MSbgR19/Engineering-Chemistry-Jain-Jain-QR.jpg",
        "card_title": "Engineering Chemistry",
        "card_subtitle": "Comprehensive Edition"
    },
    {
        "id": 6,
        "title": "Foundations of Data Science",
        "author": "Avrim Blum",
        "isbn": "9386279800",
        "released": "1st January 2013",
        "publisher": "Cambridge University",
        "format": "Hardcover",
        "main_image_url": "https://i.ibb.co/bsGLVwd/14.jpg",
        "qr_code_url": "https://i.ibb.co/rxjb7PP/3.png",
        "card_title": "Data Science Foundations",
        "card_subtitle": "An In-Depth Introduction"
    },
    {
        "id": 7,
        "title": "Genetics and Molecular Biology",
        "author": "Robert Schleif",
        "isbn": "0801846749",
        "released": "1 January 1993",
        "publisher": "The Johns Hopkins University Press",
        "format": "Hardcover",
        "main_image_url": "https://i.ibb.co/GxpxvXK/16.jpg",
        "qr_code_url": "https://i.ibb.co/3B9Zzhh/6.png",
        "card_title": "Genetics & Molecular Biology",
        "card_subtitle": "Comprehensive Guide"
    },
    {
        "id": 8,
        "title": "Learn Bengali",
        "author": "Hudson",
        "isbn": "0340057726",
        "released": "January 1, 1973",
        "publisher": "Hodder Arnold H&S",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/F06WVh1/15.jpg",
        "qr_code_url": "https://i.ibb.co/F7Q3bNR/4.png",
        "card_title": "Learn Bengali",
        "card_subtitle": "A Beginner's Guide"
    },
    {
        "id": 9,
        "title": "Principles of Management",
        "author": "Godfred",
        "isbn": "9788180522758",
        "released": "May 21, 2018",
        "publisher": "Margham Publications",
        "format": "Paperback",
        "main_image_url": "https://i.ibb.co/KwhDNV4/12.jpg",
        "qr_code_url": "https://i.ibb.co/FXkh1ZY/5.png",
        "card_title": "Management Principles",
        "card_subtitle": "Foundational Concepts"
    },
    {
        "id": 10,
        "title": "The Story of Modern France",
        "author": "H.A. Guerber",
        "isbn": "1015416845",
        "released": "August 19, 2017",
        "publisher": "Andesite Press",
        "format": "Paperback",
        "main_image_url": "https://cdn.waterstones.com/bookjackets/large/9781/7898/9781789876116.jpg",
        "qr_code_url": "https://i.ibb.co/tqXQVKQ/7.png",
        "card_title": "Modern France History",
        "card_subtitle": "Comprehensive Overview"
    }
]



########################################
""" Authentication and Authorization """

# Decorator for routes that require authentication
def auth_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated
        if 'user' not in session:
            return redirect(url_for('login'))
        
        else:
            return f(*args, **kwargs)
        
    return decorated_function


@app.route('/auth', methods=['POST'])
def authorize():
    token = request.headers.get('Authorization')
    if not token or not token.startswith('Bearer '):
        return "Unauthorized", 401

    token = token[7:]  # Strip off 'Bearer ' to get the actual token

    try:
        decoded_token = auth.verify_id_token(token) # Validate token here
        session['user'] = decoded_token # Add user to session
        return redirect(url_for('dashboard'))
    
    except:
        return "Unauthorized", 401


#####################
""" Public Routes """
TOTAL_APPLICANTS = 1000

@app.route('/form')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        rank = int(request.form['rank'])
        college = request.form['college']
        department = request.form['department']

        # Load the CSV file
        seats_df = pd.read_csv('D:/flask website/Flask-Firebase-Template/College_Seats_Data (1).csv')

        # Filter for the selected college and department
        filtered_data = seats_df[(seats_df['College Name'] == college) & 
                                 (seats_df['Department Name'] == department)]

        if filtered_data.empty:
            return render_template('predict.html', error='Invalid college or department selection.')

        # Extract seat information
        department_seats = filtered_data.iloc[0]['Seats Available']
        total_seats = seats_df['Seats Available'].sum()

        # Calculate seat percentage
        seat_percentage = department_seats / total_seats

        # Calculate rank scaling factor
        rank_scaled = 1 - (rank - 1) / (TOTAL_APPLICANTS - 1)

        # Create a dataframe for the input
        candidate = pd.DataFrame({'Rank_Scaled': [rank_scaled], 'Seat_Percentage': [seat_percentage]})

        # Predict probability
        probability = model.predict_proba(candidate)[:, 1][0]

        return render_template('predict.html', probability=f"{probability * 100:.2f}%")
    except Exception as e:
        return render_template('predict.html', error=str(e))

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/books', methods=['GET', 'POST'])
def book_search():
    query = request.args.get('query', '').lower()  # Get the search query from the URL
    filtered_books = books

    # Filter the books based on the query
    if query:
        filtered_books = [
            book for book in books
            if query in book["title"].lower() or query in book["author"].lower()
        ]
    
    return render_template('book_card.html', books=filtered_books, query=query)
# @app.route('/card')
# def card():
#     # Pass dynamic content to the template if needed
#     return render_template('card_reveal.html', 
#                            main_image_url="https://m.media-amazon.com/images/I/61Iz2yy2CKL.jpg",
#                            qr_code_url="https://api.qrserver.com/v1/create-qr-code/?data=https://example.com&size=100x100",
#                            card_title="Decoding DSA",
#                            card_subtitle="Second Edition")
@app.route("/showcard/<int:book_id>")
def show_card(book_id):
    # Find the book in the local list using the book_id
    book = next((b for b in books if b["id"] == book_id), None)
    if not book:
        return "Book not found", 404
    return render_template("card_reveal.html", book=book)
   

@app.route('/login')
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('login.html')
PISTON_URL = "https://emkc.org/api/v2/piston/execute"

# Route to render the code editor
@app.route('/Code')
def code_editor():
    return render_template('codeEditor.html')

# Route to execute the code via Piston API
@app.route('/execute', methods=['POST'])
def execute_code():
    data = request.json
    language = data.get('language')
    code = data.get('code')

    # Payload for Piston API
    payload = {
        "language": language,
        "version": "*",
        "files": [{"content": code}]
    }

    try:
        # Send request to Piston API
        response = requests.post(PISTON_URL, json=payload)
        response_data = response.json()

        # Return the output to the frontend
        return jsonify({
            "output": response_data.get('run', {}).get('output', 'Error executing code').strip(),
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/signup')
def signup():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('signup.html')


@app.route('/reset-password')
def reset_password():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    else:
        return render_template('forgot_password.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/logout')
def logout():
    session.pop('user', None)  # Remove the user from session
    response = make_response(redirect(url_for('login')))
    response.set_cookie('session', '', expires=0)  # Optionally clear the session cookie
    return response


##############################################
""" Private Routes (Require authorization) """

@app.route('/dashboard')
@auth_required
def dashboard():

    return render_template('dashboard.html')






if __name__ == '__main__':
    app.run(debug=True)