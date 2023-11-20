import os
import time
import glob

import pandas as pd

# Import selenium webdriver
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

downloads = r'C:\Users\arora\Downloads'


def upload_csv(split_path):
    
    glob_pth = r"C:\Users\arora\Desktop\cluster"
    # Create a driver object
    driver = webdriver.Edge()

    # Navigate to the web page
    driver.get("http://lexhub.org/wlt/lexica.html")
    
    button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "add_doc"))
    )
    button.click()
    
    radio = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "opt_input_csv"))
    )
    radio.click()
    
    csv_upload = driver.find_element(By.ID, "CSVFile")
    csv_upload.send_keys(os.path.join(glob_pth, split_path))
    
    csv_submit = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "upload_csv"))
    )
    csv_submit.click()
    
    # AgeGender
    AgeGender = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "AgeGender"))
    )
    AgeGender.click()
    
    terms_check = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "terms_check"))
    )
    terms_check.click()
    
    submit_all = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.ID, "submit_all"))
    )
    submit_all.click()
    
    download_to_csv = WebDriverWait(driver, 30).until(
        EC.element_to_be_clickable((By.ID, "download_to_csv"))
    )
    download_to_csv.click()
    time.sleep(5)
    
    try:
        driver.close()
    except:
        pass
    
    # files = [x for x in os.listdir(downloads) if x.endswith(".csv")]
    files = glob.glob(downloads + "\*csv")
    newest = max(files , key = os.path.getctime)
    csv_path = newest
    
    return os.path.join(downloads, csv_path)


def extract_integer(filename):
    return int(filename.split('.')[0].split('_')[1])

if __name__ == '__main__':

    # Close the driver
    # time.sleep(10)
    get_lexhub_scores = False
    data_pth = r"datasets\paraphrases-csv"
    par_csv = r"par_gender_infer_all.csv"
    
    labeled_dir = r"datasets\paraphrases-csv\labeled"
    
    
    paraphrase_csv = pd.read_csv(os.path.join(data_pth, par_csv))
    print(paraphrase_csv.head())
    
    if get_lexhub_scores:
        split_files = sorted(os.listdir(os.path.join(data_pth, "split")),  key=extract_integer)
        for file in split_files:
            print("Splitname: ", file)
            lex_csv_out = upload_csv(os.path.join(data_pth, "split", file))
            print("###################: ", lex_csv_out)
            df = pd.read_csv(lex_csv_out)
            df.to_csv(os.path.join(labeled_dir, file), index=False)
    else:
        all_labeled_df = pd.DataFrame(columns=['Row ID', 'value'])
        l_split_files = sorted(os.listdir(os.path.join(data_pth, "labeled")),  key=extract_integer)
        for file in l_split_files:
            print("Splitname: ", file)
            df = pd.read_csv(os.path.join(labeled_dir, file), header=None)
            df.columns = ['Row ID', 'score', 'type', 'value']
            df = df[df['type']=='GENDER'].sort_values(by=['Row ID'])
            df = df[['Row ID', 'value']]
            print(df.head())
            all_labeled_df = pd.concat([all_labeled_df, df], ignore_index=True, axis=0)
            
        print(all_labeled_df.head())
        paraphrase_csv = paraphrase_csv.join(all_labeled_df.set_index('Row ID'), on='Row ID')

        paraphrase_csv.to_csv(os.path.join(data_pth, par_csv.split(".")[0]+"labeled.csv"))
    
    # # upload_csv("path")
    # time.sleep(10)
    

    
