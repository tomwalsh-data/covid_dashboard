# covid_dashboard
plotly/dash app to visualize covid in England

build & run
see https://stackoverflow.com/questions/61194881/docker-container-run-locally-didnt-send-any-data
docker build -t covid_dashboard .
docker run -p 8050:8050 covid_dashboard
access on localhost:8050

`covid_dashboard.py` contains the main code to visualize the spatial distribution of cases, deaths, and vaccinations across England at a chosen date. 
Time series for interactivly selected councils are show on the right. Data may be daily, weekly (7-day rolling average), or cumulative.

Three .csv files are included for demonstration purposes: this avoids the need to access the gov.uk api end-point repeatedly (which can be a bit hit-and-miss on their end).

Finally, `get_data.py` is imported to the main file and, if called in the main code, fetches the most recent data from the government site.

Note: the dashboard requires plotly v.5+ and dash 2.0.0+
