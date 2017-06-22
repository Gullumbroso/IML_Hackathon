# Imports the Google Cloud client library
from google.cloud import language

# from oauth2client.client import GoogleCredentials
#
# credentials = GoogleCredentials.get_application_default()

# Instantiates a client
language_client = language.Client()

# The text to analyze
text = "i'm sad!"
document = language_client.document_from_text(text)

# Detects the sentiment of the text
sentiment = document.analyze_sentiment().sentiment

print('Text: {}'.format(text))
print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))