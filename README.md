# Disaster Response Pipeline Project
### Table of Contents

1. [Project Motivation](#motivation)
2. [File Descriptions](#files)
3. [Usuage](#Usuage)
4. [License](#license)
 
## Project Motivation <a name="motivation"></a>
This is an application designed to categorize emergenecy messages to their corresponding categry the datset is provided by Figure Eight.
The motivation behind this is to allow related facilites to react as quickly as possible to emergency messages is if it is classifed correctly then that will allow the corrsponding department to respond quicker and thus saving lives 
## File Descriptions  <a name="files"></a>
The data directory has a file process_data.py which cleans the data, DisasterResponse.db the database which has the cleaned data, disaster_messages.csv and disaster_categories.csv which are the raw data processed 
The models directory has a train_classifier.py which trains the ML model to classify the data and classfier.zip  which is the trained model but zipped as it was too big to upload directly to github you would have to unzip it and name it as classifier.pk for it work with web app correctly
The app directory has the HTML templates and the run.py which visulize the results into a web app

Link to github repo https://github.com/youssefmecky96/Classifaction_for_Disaster_Response_messages

## Usuage <a name="Usuage"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
## License <a name="license"></a>

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
