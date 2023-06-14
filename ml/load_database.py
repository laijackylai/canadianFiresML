import pandas as pd
import sqlite3


def read_fire_data():
    '''
    read fire data
    '''
    # Provide the fire file path
    file_path = './data/fire/NFDB_point_20220901.txt'

    # Read the CSV file into a DataFrame
    fire_df = pd.read_csv(file_path, delimiter=',')

    print('Read fire data')
    return fire_df[['FIRE_ID', 'SRC_AGENCY', 'LATITUDE', 'LONGITUDE', 'YEAR', 'MONTH', 'DAY', 'CAUSE', 'SIZE_HA', 'OUT_DATE']]


if __name__ == "__main__":
    fire_data = read_fire_data()

    # Connect to the SQLite database
    conn = sqlite3.connect('data.db')

    # Write the DataFrame to a table in the database
    fire_data.to_sql('fire', conn, if_exists='replace', index=False)

    # Close the database connection
    conn.close()
    print('added fire data to database')
