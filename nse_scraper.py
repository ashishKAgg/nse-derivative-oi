import requests
import os, time
import pandas as pd
import json, warnings
warnings.filterwarnings("ignore")
from datetime import datetime

indexSymbol = os.getenv('index_symbol', 'nse50_opt')
indexName = os.getenv('index_name', 'nifty50')

nseUrl = "https://www.nseindia.com/market-data/equity-derivatives-watch"
nseOptionsUrl = f"https://www.nseindia.com/api/liveEquity-derivatives?index={indexSymbol}"

headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36", 
          "referer": "https://www.nseindia.com/market-data/equity-derivatives-watch"}

outputDir = "C:\\xyz\\optionChart\\nifty\\optionsOI\\"
prefixPath = outputDir + "optionsoi-"

repoDirectory = f'optionOIData/{indexName}'
repoPrefixPath = f'{repoDirectory}/{indexName}-'

def createDirIfNotExists(csvPath):
    dirPath = csvPath[:csvPath.rfind("\\")]
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)


def fetchOptionsFromNSE():
    #Handle Cookies
    with requests.Session() as sess:
        sess.get(nseUrl, headers=headers, verify=False)
        resp = sess.get(nseOptionsUrl, headers=headers, verify=False)

    if resp.status_code == 200:
        print('Received 200 response from NSE')
        data = json.loads(resp.text)
        #Transform json to DataFrame
        df = pd.DataFrame(data['data'])
        #Added DateTime column
        df["datetime"] = datetime.now()
        #Remove unused columns
        removeUnusedFeatures(df)
        #Select only current and next week data
        #print(df.head())
        df = selectTwoWeeksData(df)
    else:
        print("Error Occurred..Status Code:"+str(resp.status_code))
    return df


def removeUnusedFeatures(df):
    del df["underlying"]
    del df["identifier"] 
    del df["instrumentType"] 
    del df["instrument"]
    return df


def selectTwoWeeksData(df):
    currWeek = int(datetime.now().isocalendar()[1])
    df["expiryDate_week"] = pd.to_datetime(df["expiryDate"], format="%d-%b-%Y").dt.isocalendar().week
    ndf = df[(df["lastPrice"] > 0) & ((df["expiryDate_week"] == currWeek) | (df["expiryDate_week"] == currWeek+1))]
    del ndf["expiryDate_week"]
    ndf.reset_index(inplace=True, drop=True)
    return ndf


def saveToFile(df, prefixPath):
    grouped = df.groupby("expiryDate")
    for name, group in grouped:
        csvPath = prefixPath + name + ".csv"
        #TODO: find file with latest counter and generate new CSVPath
        #csvPath = checkAndCreateNew(csvPath)
        if os.path.exists(csvPath):
            group.to_csv(csvPath, mode="a", index=False, header=False)
        else:
            group.to_csv(csvPath, mode="a", index=False)


def saveToRepo(df, directory, repoPrefixPath):
    # Create the folder if it doesn't exist (equivalent to 'mkdir -p')
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    grouped = df.groupby("expiryDate")
    for name, group in grouped:
        csvPath = repoPrefixPath + name + ".csv"
        if os.path.exists(csvPath):
            print(f'File exists, appending data into it. {csvPath}')
            group.to_csv(csvPath, mode="a", index=False, header=False)
        else:
            print(f'Creating new file: {csvPath}')
            group.to_csv(csvPath, mode="a", index=False)
        print(f'Successfully created/appended data in file: {csvPath}')


def checkAndCreateNew(csvPath):
    newCsvPath = csvPath
    
    if os.path.exists(csvPath):
        maxFileSize = 1000
        currFileSizeInKB = os.stat(csvPath).st_size/1000
        delimiter = '_'
        
        if currFileSizeInKB > maxFileSize:
            nameExtSplit = os.path.splitext(csvPath)
            filename = nameExtSplit[0]
            lastOcc = filename.rfind(delimiter)
            if lastOcc == -1:
                counter = 1
                lastOcc = len(filename)
            else:
                counter = int(filename[lastOcc + 1:]) + 1

            newCsvPath = filename[:lastOcc] + delimiter + str(counter) + nameExtSplit[1]
            print("new file created:"+newCsvPath)
    return newCsvPath


if __name__ == "__main__":
    print('Reading data from NSE')
    df = fetchOptionsFromNSE()
    print('Fetched data from NSE, saving it in CSV file')
    # saveToFile(df,prefixPath)
    saveToRepo(df, repoDirectory, repoPrefixPath)
