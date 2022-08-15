from datetime import datetime as dt
import numpy as np
import pandas as pd
import requests

def get_data(verbose=False):
    """
    Download data from the gov.uk coronavirus dashboard back to the start of the pandemic
    ARGS:
        verbose (bool): print out messages tracking the success of requests

    RETURN:
        three pandas dataframes: daily figures (nominal); weekly averages (rates); cumulative figures (nominal)
        for cases, deaths, and vaccination (1st dose & second dose)
    """

    def get_data(metrics, max_attempts=5):
        """
        Define the API endpoints for each of the three dataframes, send the requests, and tidy up the columns
        ARGS:
                 metrics (str): determines the endpoint URL 
            max_attempts (int): maximum number of attempts to download the data
                                before aborting (site can be a little buggy)

        RETURN:
            cleaned up dataset on successs
            None otherwise
        """

        def send_request(payload):
            """
            Send the request
            ARGS:
                payload (dict): passed to requests.get()

            RETURN:
                raw .json
            """

            root = "https://coronavirus.data.gov.uk/api/v2/data?areaType=ltla"

 
            r = requests.get(root, params=payload)
            if r.status_code == 200:
                # success :)
                if verbose:
                    print("Data successfully retrieved after {} attempt{}".format(i, "s" if i != 1 else ''))
                    
                data =  pd.DataFrame().from_dict(r.json()['body'])

            else:
                # failure :(
                if verbose:
                    print("Failed to retrieve data. The website is tempermental. Check that it's working")
                    
                data = None

            return data

        # get 1st DataFrame
        if metrics == "cumulative":
            payload = {
                'metric': [
                    'cumCasesBySpecimenDate',
                    'cumDeaths28DaysByDeathDate',
                    'cumVaccinationFirstDoseUptakeByVaccinationDatePercentage',
                    'cumVaccinationSecondDoseUptakeByVaccinationDatePercentage'],
                'format': 'json'
            }

            data = send_request(payload)


            # clean up headers and dtypes, set index
            if data is not None:
                data.drop('areaType', axis=1, inplace=True)
                data.rename(
                    columns = {
                        'areaCode': 'area_code',
                        'areaName': 'area_name', 
                        'date': 'date',
                        'cumVaccinationFirstDoseUptakeByVaccinationDatePercentage': 'dose_1',
                        'cumVaccinationSecondDoseUptakeByVaccinationDatePercentage': 'dose_2',
                        'cumDeaths28DaysByDeathDate': 'deaths',
                        'cumCasesBySpecimenDate': 'cases'
                    }, inplace=True
                )

                data['date'] = pd.to_datetime(data['date'])
                data.set_index('area_code', inplace=True)

        # get 2nd DataFrame
        elif metrics == "daily_rates":
            payload = {
                'metric': [
                    'newCasesBySpecimenDate',
                    'newDeaths28DaysByDeathDate',
                    'newPeopleVaccinatedFirstDoseByVaccinationDate',
                    'newPeopleVaccinatedSecondDoseByVaccinationDate'],
                'format': 'json'
            }

            data = send_request(payload)

            # clean up headers and dtypes, set index
            if data is not None:

                data.drop('areaType', axis=1, inplace=True)  
                data.rename(
                    columns = {
                        'areaCode': 'area_code',
                        'areaName': 'area_name', 
                        'date': 'date',
                        'newCasesBySpecimenDate': 'cases',
                        'newDeaths28DaysByDeathDate': 'deaths', 
                        'newPeopleVaccinatedFirstDoseByVaccinationDate': 'dose_1',
                        'newPeopleVaccinatedSecondDoseByVaccinationDate': 'dose_2'
                    }, inplace=True
                )

                data['date'] = pd.to_datetime(data['date'])
                data.set_index('area_code', inplace=True)

        # get 3rd DataFrame
        elif metrics == "weekly_rates":
            payload = {
                'metric': [
                    'newCasesBySpecimenDateRollingRate',
                    'newDeaths28DaysByDeathDateRollingRate',
                    'newPeopleVaccinatedFirstDoseByVaccinationDate',
                    'newPeopleVaccinatedSecondDoseByVaccinationDate'],
                'format': 'json'
            }

            data = send_request(payload)

            # clean up headers and dtypes, set index
            if data is not None:
                data.drop('areaType', axis=1, inplace=True)
                data.rename(
                    columns = {
                        'areaCode': 'area_code',
                        'areaName': 'area_name', 
                        'date': 'date',
                        'newPeopleVaccinatedFirstDoseByVaccinationDate': 'dose_1',
                        'newPeopleVaccinatedSecondDoseByVaccinationDate': 'dose_2', 
                        'newCasesBySpecimenDateRollingRate': 'cases',
                        'newDeaths28DaysByDeathDateRollingRate': 'deaths'
                    }, inplace=True
                )

                data['date'] = pd.to_datetime(data['date'])
                data.set_index('area_code', inplace=True)

        # invalid agrs were passed: this should never fire
        else:
            pass

        return data

        
    # top-level function calls
    daily = get_data('daily_rates')
    weekly = get_data('weekly_rates')
    cumulative = get_data('cumulative')

    # calculate 7-day rolling average for vaccine doses (not published at source)
    for code in weekly.index.unique():
        temp = weekly.loc[code, ['date', 'dose_1', 'dose_2']]
        temp = temp.rolling(window=7, center=True, on='date').mean()
        weekly.loc[code, 'dose_1'] = temp['dose_1']
        weekly.loc[code, 'dose_2'] = temp['dose_2']

    print("Datasets retreived successfully")
    return daily, weekly, cumulative


if __name__ == "__main__":
    pass

