import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import spacy
import argparse
import sqlite3
from dateutil import parser as date_parser


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


def load_data():
    """
    load fire data from sqlite database
    """
    # Connect to the SQLite database
    conn = sqlite3.connect('./ml/data.db')

    # Query to fetch the data from the table (replace 'table_name' with the actual table name)
    query = "SELECT * FROM fire"

    # Load the data into a DataFrame
    f_data = pd.read_sql_query(query, conn)

    # Close the database connection
    conn.close()
    return f_data


def is_date(date_str):
    '''
    check whether string is date
    '''
    try:
        date_parser.parse(date_str)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    # * setup
    # Load NLTK stopwords
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    nlp = spacy.load('en_core_web_sm')

    # # Example DataFrame
    # # data = {
    # #     'PROVINCE_CODE': ['AB', 'BC', 'AB', 'ON', 'BC'],
    # #     'LOCAL_YEAR': [2020, 2020, 2021, 2022, 2022],
    # #     'LOCAL_MONTH': [6, 7, 8, 9, 10],
    # #     'MEAN_TEMPERATURE': [25.0, 28.5, 30.2, 22.7, 26.4],
    # #     'CAUSE': ['Lightning', 'Human', 'Human', 'Lightning', 'Human'],
    # #     'SIZE_HA': [100, 50, 200, 150, 75],
    # #     'CHANCE_OF_FIRE': [0.8, 0.6, 0.9, 0.7, 0.5]
    # # }

    # # load fire data
    # data = load_data()

    # # * data preprocessing
    # df = pd.DataFrame(data)

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

    # # Replace province codes with province names, ignoring the ones not found
    # df['PROVINCE_CODE'] = [province_mapping.get(code, code)
    #                        for code in df['SRC_AGENCY']]

    # # Preprocess the dates
    # df['YEAR'] = pd.to_numeric(
    #     df['YEAR'], errors='coerce')  # Convert to numeric
    # df['MONTH'] = pd.to_numeric(
    #     df['MONTH'], errors='coerce')  # Convert to numeric
    # df['DAY'] = pd.to_numeric(
    #     df['DAY'], errors='coerce')  # Convert to numeric

    # # Handle zero values in DAY and MONTH columns
    # df.loc[df['DAY'] == 0, 'DAY'] = 1  # Change 0 values to 1 in DAY column
    # df = df[df['MONTH'] != 0]  # Drop rows with 0 values in MONTH column

    # # Merge 'LOCAL_YEAR' and 'LOCAL_MONTH' into a single date column
    # df['DATE'] = pd.to_datetime(df['YEAR'].astype(
    #     str) + '-' + df['MONTH'].astype(str) + '-01')

    # # Drop unnecessary columns
    # # df.drop(['YEAR', 'MONTH', 'DAY'], axis=1, inplace=True)

    # # Convert CAUSE column to lowercase
    # df['CAUSE'] = df['CAUSE'].fillna('NULL')
    # df['CAUSE'] = df['CAUSE'].str.lower()

    # # Text preprocessing using NLTK
    # stop_words = set(stopwords.words('english'))

    # # Preprocess the 'CAUSE' column
    # df['CAUSE'] = df['CAUSE'].apply(lambda x: ' '.join(
    #     [word for word in word_tokenize(x) if word.lower() not in stop_words]))

    # # Preprocess the DataFrame
    # df['text'] = 'fire: date ' + df['DATE'].astype(str) + ' province code ' + df['PROVINCE_CODE'] + ' cause ' + df['CAUSE'].astype(
    #     str) + ' size of fire in hectors ' + df['SIZE_HA'].astype(str)
    # df['text'] = df['text'].apply(lambda x: ' '.join(
    #     [word.lower() for word in word_tokenize(x)]))

    # conn = sqlite3.connect('./ml/nltk.db')

    # # Write the DataFrame to a table in the database
    # df[['DATE', 'PROVINCE_CODE', 'LATITUDE', 'LONGITUDE',  'CAUSE', 'SIZE_HA', 'OUT_DATE',
    #     'YEAR', 'MONTH', 'DAY', 'text']].to_sql('fire', conn, if_exists='replace', index=False)

    # conn.close()

    # print('done')
    # exit()

    # * load data from sqlite db
    # conn = sqlite3.connect('./ml/nltk.db')

    # cursor = conn.cursor()
    # cursor.execute('SELECT * FROM nltk')
    # rows = cursor.fetchall()

    # columns = [desc[0] for desc in cursor.description]
    # df = pd.DataFrame(rows, columns=columns)

    # cursor.close()
    # conn.close()

    # * load data from file
    file_path = './ml/nltk.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, delimiter=',', engine='python')

    # * User input query
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="Search query", required=True)
    parser.add_argument("--strategy", help="Search query", required=True)
    args = parser.parse_args()

    # Extract the query from command-line arguments
    user_query = args.query
    # user_query = input("Enter your search query: ")

    # * match dates
    user_query = match_dates(user_query)

    # * check for province mappings
    user_query = check_province(user_query, province_mapping)

    # * strategy: either absolute or optimistic
    strategy = args.strategy
    # strategy = "optimistic"

    if strategy == "Optimistic":
        # Search the DataFrame
        results = search_dataframe_optimistic(user_query, df)
    if strategy == "Absolute":
        results = search_dataframe_absolute(user_query, df)

    # * write results in csv
    results.to_csv('ml/query.csv', index=False)

    # * write results in superbase

    # * Display the results
    if not results.empty:
        # conn = sqlite3.connect('./ml/query.db')
        # results.to_sql('query', conn, if_exists='replace', index=False)
        # conn.close()
        print('success')
    else:
        print("failed")
