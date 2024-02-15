npm install package-name
import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors


# <p align="center">Title</p>
  
file_path = r"C:\Users\solea\Desktop\Hackathon\okcupid_profiles.csv"
df = pd.read_csv(file_path)

file_path2 = r"C:\Users\solea\Desktop\Hackathon\prete.csv"
prete = pd.read_csv(file_path2)
    
