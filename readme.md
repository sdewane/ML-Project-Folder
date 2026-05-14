1. **Download the dataset here**
 https://drive.google.com/drive/folders/13GTbJUsUwlzMJt1fEdBIQn6Cs1hFfC1_?usp=drive_link

**Make sure to put the datasets into a folder named "data" within the ML-Project-Folder** or change where the code looks for the datasets by changing the pd.read_csv('dataset path in directory') in the data_prep.py and the variable data_path in main.
If you are cloning the ML-Project-Folder into another folder, I would recommend moving the ML-Project-Folder so it can exist as a standalone folder instead of being embedded in another folder. If not its okay, just make sure to run step 5.
**If you just want to see the finalized dataset and run the program, you only need to download 'final_cleaned_market_data.csv'.** 
**If you want to see how the system actually collects and cleans the data, download all the .csv files EXCEPT for 'final_cleaned_market_data.csv'.** To save time, main won't re-make a 'final_cleaned_market_data.csv' if it detects it in the 'data' folder.

2. **Create a Virtual Environment**
   - Mac/Linux: `python3 -m venv .venv`
   - Windows: `python -m venv .venv`

3. **Activate the Environment**
   - Mac/Linux: `source .venv/bin/activate`
   - Windows: `.\.venv\Scripts\activate`

4. **Install Dependencies**
   `pip install -r requirements.txt`

5. **Once data sets are saved in a folder named 'data' in ML-Project-Folder**
   `cd .\ML-Project-Folder\`  
   `python main.py`
