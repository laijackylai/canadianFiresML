from flask import Flask, jsonify, request
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
from dateutil import parser as date_parser

app = Flask(__name__)


@app.route("/")
def default():
    '''
    default route
    '''
    return "Search Canadian Fire Data with NLTK (/data)"


@app.route("/data", methods=["GET"])
def get_data():
    '''
    get nltk data
    '''
    strategy = request.args.get("strategy")
    query = request.args.get("query")

    if not strategy or not query:
        return jsonify({"message": "Both strategy and query parameters are required"}), 400

    if strategy not in ["Absolute", "Optimistic"]:
        return jsonify({"message": "Invalid strategy"}), 400

    if query == "":
        return jsonify({"message": "Invalid query"}), 400

    # * init
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    nlp = spacy.load('en_core_web_sm')
    province_mapping = {
        'AB': 'Alberta',
        'BC': 'British Columbia',
        'MB': 'Manitoba',
        'NB': 'New Brunswick',
        'NL': 'Newfoundland and Labrador',
        'NS': 'Nova Scotia',
        'NT': 'Northwest Territories',
        'NU': 'Nunavut',
        'ON': 'Ontario',
        'PE': 'Prince Edward Island',
        'QC': 'Quebec',
        'SK': 'Saskatchewan',
        'YT': 'Yukon'
    }
    user_query = query

    def lemmatize(text):
        """
        Lemmatize a text using spaCy's lemmatizer.
        """
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc]
        if len(lemmas) > 0:
            return lemmas[0]
        else:
            return ""

    def parse_date(date_str):
        """
          Parse a date string into a standardized format "YYYY-MM-DD".
        """
        patterns = [
            r'(\d{4}) (\d{2})',       # yyyy mm
            r'(\d{2}) (\d{4})',       # mm yyyy
            r'(\w{3}) (\d{4})',       # mmm yyyy
            r'(\d{4}) (\w{3})',       # yyyy mmm
            r'(\d{2})\. (\d{4})',     # mm. yyyy
            r'(\d{2}), (\d{4})',      # mm, yyyy
            r'(\w{3})\. (\d{4})',     # mmm. yyyy
            r'(\w{3}), (\d{4})',      # mmm, yyyy
            r'(\d{2})/(\d{4})',       # mm/yyyy
            r'(\d{4})-(\d{2})',       # yyyy-mm
            r'(\d{2})-(\d{4})',       # mm-yyyy
            r'(\w{3})-(\d{4})',       # mmm-yyyy
            r'(\d{4})-(\w{3})',       # yyyy-mmm
            r'(\d{2})/(\d{2})/(\d{4})'  # mm/dd/yyyy
        ]

        for pattern in patterns:
            match = re.match(pattern, date_str)
            if match:
                if len(match.groups()) == 2:
                    year = int(match.group(2))
                    month = match.group(1)
                    return f"{year}-{month}-01"
                elif len(match.groups()) == 3:
                    if match.group(1).isdigit() and match.group(2).isdigit():
                        year = int(match.group(3))
                        month = int(match.group(1))
                    else:
                        year = int(match.group(2))
                        month = match.group(1)
                    return f"{year}-{month:02d}-01"

        return None

    def is_date(date_str):
        '''
        check whether string is date
        '''
        try:
            date_parser.parse(date_str)
            return True
        except ValueError:
            return False

    def match_dates(uq):
        """
        Match and replace date patterns in a user query with parsed dates.
        """

        # Define a regular expression pattern to match dates
        patterns = [
            r'\d{4} \d{2}',       # yyyy mm
            r'\d{2} \d{4}',       # mm yyyy
            r'\w{3} \d{4}',       # mmm yyyy
            r'\d{4} \w{3}',       # yyyy mmm
            r'\d{2}\. \d{4}',     # mm. yyyy
            r'\d{2}, \d{4}',      # mm, yyyy
            r'\w{3}\. \d{4}',     # mmm. yyyy
            r'\w{3}, \d{4}',      # mmm, yyyy
            r'\d{2}/\d{4}',       # mm/yyyy
            r'\d{4}-\d{2}',       # yyyy-mm
            r'\d{2}-\d{4}',       # mm-yyyy
            r'\w{3}-\d{4}',       # mmm-yyyy
            r'\d{4}-\w{3}'        # yyyy-mmm
        ]

        matches = []
        for pattern in patterns:
            matches += re.findall(pattern, uq)

        # Parse and replace the dates in the user query
        for match in matches:
            parsed_date = parse_date(match)
            if(parsed_date is None):
                continue
            else:
                uq = uq.replace(match, parsed_date)
        return uq

    def check_province(uq, province_mapping):
        """check for province shorthands and replace them

        Args:
            uq (_type_): user_query
            province_mapping (_type_): province name mappings

        Returns:
            _type_: parsed user query
        """
        query_words = uq.split()
        for i, word in enumerate(query_words):
            if word in province_mapping:
                query_words[i] = province_mapping[word]
        uq = ' '.join(query_words)
        return uq

    def search_dataframe_optimistic(query, dataframe):
        """
        Search a dataframe for rows that match the most queries.
        """
        tokenized_query = word_tokenize(query)
        preprocessed_query = []
        for word in tokenized_query:
            word_lower = word.lower()
            if is_date(word_lower):
                preprocessed_query.append(word_lower)
            elif word_lower not in stop_words:
                preprocessed_query.append(lemmatize(word_lower))

        search_results = dataframe
        result_counts = {}
        for word in preprocessed_query:
            matches = search_results[search_results['text'].str.contains(
                word, case=False)].index
            for match in matches:
                result_counts[match] = result_counts.get(match, 0) + 1
        if result_counts:
            max_count = max(result_counts.values())
            result_indices = [idx for idx,
                              count in result_counts.items() if count == max_count]
            return search_results.loc[result_indices, ['ID', 'DATE', 'PROVINCE_CODE', 'LATITUDE', 'LONGITUDE',  'CAUSE', 'SIZE_HA', 'OUT_DATE', 'YEAR', 'MONTH', 'DAY']]
        else:
            return pd.DataFrame(columns=['ID', 'DATE', 'PROVINCE_CODE', 'LATITUDE', 'LONGITUDE', 'CAUSE', 'SIZE_HA', 'OUT_DATE', 'YEAR', 'MONTH', 'DAY'])

    def search_dataframe_absolute(query, dataframe):
        """
        Search a dataframe for rows that match all queries.
        """
        tokenized_query = word_tokenize(query)

        preprocessed_query = []
        for word in tokenized_query:
            word_lower = word.lower()
            if is_date(word_lower):
                preprocessed_query.append(word_lower)
            elif word_lower not in stop_words:
                preprocessed_query.append(lemmatize(word_lower))

        search_results = dataframe
        for word in preprocessed_query:
            search_results = search_results[search_results['text'].str.contains(
                word, case=False)]
        return search_results[['ID', 'DATE', 'PROVINCE_CODE', 'LATITUDE', 'LONGITUDE', 'CAUSE', 'SIZE_HA', 'OUT_DATE', 'YEAR', 'MONTH', 'DAY']]

    # * load data
    file_path = './nltk.csv'
    df = pd.read_csv(file_path, delimiter=',', engine='python')

    # * match dates
    user_query = match_dates(user_query)

    # * check for province mappings
    user_query = check_province(user_query, province_mapping)

    if strategy == "Optimistic":
        # Search the DataFrame
        results = search_dataframe_optimistic(user_query, df)
        return results.to_json()
    if strategy == "Absolute":
        results = search_dataframe_absolute(user_query, df)
        return results.to_json()

    return []


# if __name__ == "__main__":
#     debug = False
#     app.run(debug=debug)
