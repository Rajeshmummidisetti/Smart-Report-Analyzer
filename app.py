#!/usr/bin/env python
# coding: utf-8

# In[2]:


# app.py
from flask import Flask, request, render_template
from crewai import Agent, Crew
from pdfminer.high_level import extract_text
import requests
from bs4 import BeautifulSoup
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import google.generativeai as genai

app = Flask(__name__)

# Define the Gemini API function to generate recommendations
def get_recommendations_with_gemini(parsed_text):
    os.environ["API_KEY"] = "GEMINI_API_KEY"
    genai.configure(api_key=os.environ["API_KEY"])

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""Please analyze the following blood test report and identify the 2 main health issue names only. 
        List them in a structured way as follows: Health Issue: [Health Condition based on the report provided] 
        Report: {parsed_text}"""
        response = model.generate_content(prompt)
        summary = response.candidates[0].content.parts[0].text
        return summary
    except Exception as e:
        print(f"Error in Gemini API request: {e}")
        return 'Error in generating recommendations'

# Define the PDF Analysis Agent
class PDFAnalysisAgent(Agent):
    def execute(self, pdf_content):
        # Save the PDF content to a temporary file for processing
        with open("temp.pdf", "wb") as temp_pdf:
            temp_pdf.write(pdf_content)
        
        text = extract_text("temp.pdf")
        parsed_data = self.analyze_pdf(text)
        
        # Clean up the temporary file
        os.remove("temp.pdf")
        
        # Send parsed text to Gemini API for summarization
        summarized_text = get_recommendations_with_gemini(parsed_data['parsed_data'])
        return {"parsed_data": summarized_text}

    def analyze_pdf(self, text):
        # Simplify the analysis and extract keywords or meaningful parts for searching
        keywords = text.split() 
        return {"parsed_data": ' '.join(keywords)}

# Define the Web Search Agent
class WebSearchAgent(Agent):
    def execute(self, parsed_data):
        print(f"WebSearchAgent received parsed data: {parsed_data}")

        # Extract specific health issues from the structured summary for searching
        health_issues = self.extract_health_issues(parsed_data['parsed_data'])

        if not health_issues:
            return []

        articles = []
        for issue in health_issues:
            query = f"Health checkups related to {issue}"
            print(f"Executing search with query: {query}")
            articles += self.search_articles(query)

        return articles

    def extract_health_issues(self, summary):
        # Extract health issues from the structured summary
        issues = []
        lines = summary.split('\n')
        for line in lines:
            if "Health Issue:" in line:
                issue = line.split("Health Issue:")[1].strip()
                issues.append(issue)
        return issues

    def search_articles(self, query):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Scrape and process search results
            articles = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'url?q=' in href:
                    article_link = href.split('url?q=')[1].split('&')[0]
                    # Check if the article link is working
                    if self.is_link_working(article_link):
                        articles.append(article_link)
            return articles if articles else ["No relevant articles found."]
        except Exception as e:
            print(f"Error during web search: {e}")
            return []

    def is_link_working(self, url):
        """Check if the link is working by sending a HEAD request."""
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200 and response.text.strip():
                return True
            else:
                return False
        except:
            return False

# Define the Recommendation Agent
class RecommendationAgent(Agent):
    def execute(self, parsed_data, articles):
        # Check if articles are found
        if not articles or articles == ["No relevant articles found."]:
            return "No relevant health articles could be found based on the report."
        
        # Generate recommendations by checking if health issues match article URLs
        recommendations = self.generate_recommendations(parsed_data['parsed_data'], articles)
        return recommendations

    def generate_recommendations(self, parsed_text, articles):
        # Extract health issues from the parsed text
        health_issues = self.extract_health_issues(parsed_text)

        # Dictionary to store the final recommendations and related URLs
        recommendations_dict = {
            "Health Issues": [],
            "Recommendations": [],
            "Related URLs": []
        }

        # Process each health issue
        for issue in health_issues:
            recommendations_dict["Health Issues"].append(issue)
            
            # Gather content from up to two related articles
            related_urls = []
            for article in articles:
                if any(word.lower() in article.lower() for word in issue.split()):
                    related_urls.append(article)
                    if len(related_urls) >= 2:  # Limit to 2 URLs
                        break

            # Generate recommendations based on the issue
            content_combined = " ".join([self.scrape_content_from_url(url) for url in related_urls if self.scrape_content_from_url(url)])
            if content_combined:
                generated_recommendations = self.generate_recommendations_from_gemini(content_combined)
                recommendations_dict["Recommendations"].extend(generated_recommendations)

            # Add related URLs (up to 2)
            recommendations_dict["Related URLs"].extend(related_urls[:2])  # Ensure only 2 URLs

        # Prepare the final output
        final_output = (
            f"Health Issues: {', '.join(recommendations_dict['Health Issues'])}\n"
            f"Recommendations:\n- " + "\n- ".join(recommendations_dict['Recommendations'][:6]) + "\n"  # Limit to 6 recommendations
            f"Related URLs:\n- " + "\n- ".join(recommendations_dict['Related URLs'][:4])  # Limit to 4 URLs
        )
        
        final_output = final_output.replace("- Here are 6 single-line health recommendations based on the provided content:\n", "")
        final_output = final_output.replace("*", "")
        return final_output if final_output else "No specific recommendations could be generated based on the URLs."

    def extract_health_issues(self, parsed_text):
        """Extract health issues from the structured summary."""
        issues = []
        lines = parsed_text.split('\n')
        for line in lines:
            if "Health Issue:" in line:
                issue = line.split("Health Issue:")[1].strip()
                issues.append(issue)
        return issues

    def scrape_content_from_url(self, url):
        """Scrape the content from a given URL."""
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Extract text content from paragraphs
                paragraphs = soup.find_all('p')
                content = " ".join([p.get_text() for p in paragraphs])
                return content
            else:
                return None
        except Exception as e:
            print(f"Error scraping content from URL {url}: {e}")
            return None

    def generate_recommendations_from_gemini(self, content_combined):
        """Generate recommendations using the Gemini API."""
        try:
            genai.configure(api_key=os.environ["API_KEY"])
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Based on the following content, generate 6 single line health recommendations:\n{content_combined}"
            response = model.generate_content(prompt)
            summary = response.candidates[0].content.parts[0].text
            recommendations = summary.split("\n")
            return [rec for rec in recommendations if rec] 
        except Exception as e:
            return [f"Error in generating recommendations: {e}"]

# Define the Email Agent
class EmailAgent(Agent):
    def execute(self, recipient_email, message):
        sender_email = " Replace with your email"
        sender_password = " # Replace with your email password"

        # Create a MIMEMultipart object
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = "Personalized Health Recommendations"

        # Attach the message to the email
        msg.attach(MIMEText(message, 'plain'))

        # Send the email
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
            server.quit()
            print(f"Email sent to {recipient_email}")
        except Exception as e:
            print(f"Failed to send email: {e}")

# Initialize agents
pdf_agent = PDFAnalysisAgent(
    role="PDF Analysis",
    goal="Extract and analyze data from the blood test PDF.",
    backstory="An AI specialized in understanding blood test reports to provide medical insights."
)

web_search_agent = WebSearchAgent(
    role="Web Search",
    goal="Find reliable articles related to the health issues in the PDF.",
    backstory="An AI skilled in finding accurate and relevant information from the web."
)

recommendation_agent = RecommendationAgent(
    role="Recommendation",
    goal="Generate personalized health recommendations based on analyzed PDF and web search results.",
    backstory="An AI trained in providing health advice based on various sources of information."
)

email_agent = EmailAgent(
    role="Email Sending",
    goal="Send personalized health recommendations to the user's email.",
    backstory="An AI that securely sends important health information to the user's email address."
)

crew = Crew(
    agents=[pdf_agent, web_search_agent, recommendation_agent,email_agent]
)


def analyze_pdf_file(file_path, recipient_email):
    with open(file_path, "rb") as pdf_file:
        pdf_content = pdf_file.read()

    # Execute the agents manually
    pdf_analysis_result = pdf_agent.execute(pdf_content)    
    web_search_result = web_search_agent.execute(pdf_analysis_result)
    recommendations = recommendation_agent.execute(pdf_analysis_result, web_search_result)
    
    # Send the recommendations via email
    email_agent.execute(recipient_email, recommendations)

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        recipient_email = request.form['email']

        if file and recipient_email:
            file_path = "temp.pdf"
            file.save(file_path)

            recommendations = analyze_pdf_file(file_path, recipient_email)

            # Clean up the temporary PDF file
            #os.remove(file_path)

            return render_template('index.html', recommendations=recommendations)

    return render_template('index.html', recommendations=None)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




