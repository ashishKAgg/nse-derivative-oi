# nse-derivative-oi
Web scrapper for NSE derivatives data. GHA workflow runs after every 5 minutes to capture the change in Options OI data for the list of index provided in the workflow matrix.
It process the data of current and upcoming Option data expiry and save it in to file named with `Expiry date`. If file already exists then it appends the data into file, otherwise creates a new file and push the changes to repository.

## Steps to add more indexes:
Currently, it supports only Nifty50. To add more indexes:
- Open `nse_option_scheduler.yaml` workflow file.
- Add new object with `symbol` and `name` as matrix item on line 17.
