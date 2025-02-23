import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    usuage puts the data from tow filepaths in a dataframe
    input: 2 file paths
    output: Dataframe with both merged on id 
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(categories,messages,on='id')
    return df
    



def clean_data(df):
    """
    usuage cleans the dataframe by creating a column for each category, drops duplicates and removes in correct values 
    input: dataframe to be cleaned
    output: cleaned dataframe
    """
    categories = df['categories'].str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing

    category_colnames = list(row.apply(lambda x: x[0:len(x)-2]))
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df.drop(['categories'],inplace=True,axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    df.drop_duplicates(inplace=True)
    #noticed that related column has value equal 2 which doesn't make sense so we remove it here
    df=df[df['related']!=2]
    return df
    


def save_data(df, database_filename):
     """
    usuage saves the dataframe in a database 
    input: dataframe and database filename
    output: no out put
    """
    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql('project_4', engine, index=False,if_exists='replace')  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
