import streamlit as st

# Custom imports
from src.app import MultiPage
from src.app import explore_infoscc_gan

# Create an instance of the app
app = MultiPage()

# Title of the main page
st.title('Galaxy Zoo generation')

# Add all your applications (pages) here
app.add_page('Explore InfoSCC-GAN', explore_infoscc_gan.app)

# The main app
app.run()
